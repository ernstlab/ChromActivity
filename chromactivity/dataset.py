from __future__ import annotations

import copy
import os
import re
from datetime import datetime
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from chromactivity.ft_mapping import get_transformer


class Dataset:
    """Version 2"""

    def __init__(
        self,
        name: str,
        cell_type: str,
        desc: str = None,
        roadmap_or_epimap: str = "roadmap",
        bed_fn: str = None,
        feat_table_fn: str = None,
        bed_df: pd.DataFrame = None,
        sample_weights: pd.Series | None = None,
        test_chromosomes: list = None,
        binarization: str = None,
        binarization_quantiles: list = None,
        label_masking: list = None,
        signal_feature_window_size: int = 1000,
        feature_resolution: int = 25,
        feature_transform_id: str = None,
    ):
        self.name = name
        self.cell_type = cell_type

        self.desc = desc

        self.sample_weights = sample_weights

        if bed_df is not None:
            assert bed_fn is None, "Use either bed_df or bed_fn, not both."

            bed_df = bed_df.copy()  # do not modify the original

            bed_df = self._check_bed(bed_df)
            assert bed_df.columns.tolist() == [
                "chrom",
                "start",
                "end",
                "region_id",
                "values",
            ]
            self.bed_df = bed_df

        self.midpoint: bool = True

        if bed_fn is not None:
            self.bed_df = self.read_dataset_bed(bed_fn)

        self._test_idx: pd.Series | None = None
        self._train_idx: pd.Series | None = None
        self.test_chromosomes: list = test_chromosomes

        self.feat_table_fn = feat_table_fn

        self.binarization_quantiles = binarization_quantiles
        self.binarization = binarization

        # a list of allowed labels, rest are dropped
        self.label_masking = label_masking

        # feature extraction parameters
        self.signal_feature_window_size: int = signal_feature_window_size
        self.feature_resolution: int = feature_resolution
        self.roadmap_or_epimap: str = roadmap_or_epimap

        # feature transform
        self.feature_transform_id: str = feature_transform_id

        self._raw_feat_df: pd.DataFrame | None = None

    def __repr__(self):
        def filter_keys(d: dict):
            """Only keep relevant keys for concise __repr__."""
            d_ = {}
            for k, v in d.items():
                if v is None:
                    continue
                if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                    continue
                if "\n" in str(v):
                    continue
                if isinstance(v, str):
                    v_ = f'"{v}"'
                else:
                    v_ = v
                d_[k] = v_
            return d_

        dd = filter_keys(self.__dict__)
        return "Dataset(" + ", ".join([f"{k}={v}" for k, v in dd.items()]) + ")"

    def __str__(self):
        return f"Dataset[{self.cell_type}]: {self.name} ({self.roadmap_or_epimap}: {self.binarization})"

    @property
    def test_chromosomes(self):
        return self._test_chromosomes

    @test_chromosomes.setter
    def test_chromosomes(self, test_chromosomes: list):
        if test_chromosomes is not None:
            self._test_chromosomes = test_chromosomes
            self._test_idx = self.bed_df["chrom"].isin(self._test_chromosomes)
            self._train_idx = ~self.bed_df["chrom"].isin(self._test_chromosomes)

    @property
    def test_idx(self):
        assert self._test_idx is not None
        return self._test_idx & ~self.labels.isna()

    @property
    def train_idx(self):
        assert self._train_idx is not None
        return self._train_idx & ~self.labels.isna()

    @property
    def values(self):
        return self.bed_df["values"]

    @property
    def labels(self) -> pd.Series[bool]:
        if self.label_masking is None:
            return self._labels
        else:
            mask = ~self._labels.isin(self.label_masking)
            return self._labels.mask(mask)

    @property
    def coord_df(self):
        """
        Includes coordinates with no assigned labels.
        """
        return self.bed_df[["chrom", "start", "end", "region_id"]]

    @property
    def raw_coord_df(self):
        return self.coord_df

    @property
    def coord_df_masked(self):
        """
        Excludes coordinates with no assigned labels, same length as X and y.
        """
        return self.coord_df.loc[~self.labels.isna(), :]

    @property
    def coord_df_train(self):
        mask = self.train_idx
        return self.coord_df.loc[mask, :]

    @property
    def coord_df_test(self):
        mask = self.test_idx
        return self.coord_df.loc[mask, :]

    @property
    def binarization(self):
        return self._binarization

    @binarization.setter
    def binarization(self, binarization):
        if binarization is None:
            if self.values.dtype == "O":
                num_classes = len(self.values.unique())
                if num_classes == 2:
                    binarization = "categorical_2"
                elif num_classes == 3:
                    binarization = "categorical_3"
                elif num_classes == 1:
                    binarization = "categorical_1"
            else:
                binarization = "continuous"

        assert binarization is not None

        if binarization == "quantile_3":
            self._labels = self._compute_3way_quantile()
        elif binarization == "quantile_2":
            self._labels = self._compute_2way_quantile()
        elif binarization == "categorical_1":
            assert len(self.values.unique()) == 1
            self._labels = self.values
        elif binarization == "categorical_2":
            assert len(self.values.unique()) == 2
            self._labels = self.values
        elif binarization == "categorical_3":
            assert len(self.values.unique()) == 3
            self._labels = self.values
        elif binarization == "continuous":
            assert self.values.dtype == float
            self._labels = self.values
        else:
            raise ValueError(f"Unknown binarization: {binarization}")

        self._binarization = binarization

    @classmethod
    def _check_bed(cls, bed_df):
        def _generate_region_id():
            region_id_prefix = ""
            return (
                region_id_prefix
                + bed_df["chrom"].map(str)
                + "_"
                + bed_df["start"].map(str)
            )

        assert (
            bed_df.index == pd.RangeIndex(start=0, stop=bed_df.shape[0], step=1)
        ).all(), "coord_df index should be contiguous for feature extraction for now"        

        bed_columns = ["chrom", "start", "end", "region_id", "values"]
        num_cols = bed_df.shape[1]
        bed_df.columns = bed_columns[:num_cols]

        if num_cols == 3:
            bed_df["region_id"] = _generate_region_id()
            bed_df.loc[:, "values"] = np.nan
        elif num_cols == 4:
            if "region_id" in bed_df.columns:
                bed_df.loc[:, "values"] = np.nan
            else:
                bed_df["region_id"] = _generate_region_id()

        assert bed_df.shape[1] == 5
        return bed_df

    def read_dataset_bed(self, bed_fn):
        df = pd.read_csv(bed_fn, sep="\t", header=None)
        df = self._check_bed(df)
        # take midpoint
        if self.midpoint:
            df["start"] = (df["start"] + df["end"]) // 2
            df["end"] = df["start"] + 1
        return df

    def _compute_3way_quantile(self):
        quantiles = self.binarization_quantiles
        assert len(quantiles) == 4
        q_repr, q_neutral_low, q_neutral_high, q_act = self.values.quantile(quantiles)

        output_label_series = pd.Series(np.nan, index=self.values.index)

        output_label_series[self.values < q_repr] = "repressive"
        output_label_series[
            self.values.between(q_neutral_low, q_neutral_high)
        ] = "neutral"
        output_label_series[self.values > q_act] = "activating"
        return output_label_series

    def _compute_2way_quantile(self):
        quantiles = self.binarization_quantiles
        assert len(quantiles) == 2
        q_repr, q_act = self.values.quantile(quantiles)

        output_label_series = pd.Series(np.nan, index=self.values.index)

        output_label_series[self.values < q_repr] = "repressive"
        output_label_series[self.values.between(q_repr, q_act)] = "neutral"
        output_label_series[self.values > q_act] = "activating"
        return output_label_series

    def extract_features(self, dump=True):

        from chromactivity.feature_extraction import FeatureExtraction

        fe = FeatureExtraction(
            window_size=self.signal_feature_window_size,
            feature_resolution=self.feature_resolution,
            roadmap_or_epimap=self.roadmap_or_epimap,
        )

        if dump:
            if self.feat_table_fn is None:
                self.feat_table_fn = self._generate_default_feat_table_fn()

            logger.info(f"Dumping raw features to {self.feat_table_fn}")
            self._raw_feat_df = fe.dump_raw_feature_table(
                self, out_fn=self.feat_table_fn
            )
        else:
            self._raw_feat_df = fe.extract_raw_features_for_dataset(self, marks="all")

    @property
    def raw_feat_df(self):
        if self._raw_feat_df is None:
            self._raw_feat_df = self._load_feat_df()

        # make sure the shapes match up: this could go wrong if bed_df is manually
        # changed, or the wrong tbl file was read
        assert self._raw_feat_df.shape[0] == self.coord_df.shape[0]
        return self._raw_feat_df

    @property
    def X(self):
        X_cols = self._columns_rfd_to_X(self.raw_feat_df.columns)
        return self.raw_feat_df.set_axis(X_cols, axis="columns")

    @property
    def X_train(self):
        return self.X.loc[self.train_idx, :]

    @property
    def X_test(self):
        return self.X.loc[self.test_idx, :]

    @property
    def y(self):
        return self.labels

    @property
    def y_train(self):
        return self.y[self.train_idx]

    @property
    def y_test(self):
        return self.y[self.test_idx]

    def _load_feat_df(self):
        assert self.feat_table_fn is not None
        try:
            logger.debug(f"Loading {self.feat_table_fn}")
            feat_df = joblib.load(self.feat_table_fn)
            assert (
                feat_df.shape[0] == self.coord_df.shape[0]
            ), "Potentially incorrect tbl file"
            return feat_df
        except FileNotFoundError as e:
            logger.error("No feature table file found, need to extract features.")
            raise e

    def _generate_default_feat_table_fn(self):
        now_str = datetime.now().strftime("%Y-%m-%d.%Hh%M")
        return f"data/processed/feature_tables/{now_str}/{self.name}.tbl"

    @classmethod
    def _columns_rfd_to_X(cls, raw_feat_df_columns: pd.MultiIndex) -> pd.Index:
        """
        Convert the raw feature dataframe columns to the X matrix columns.
        """

        def offset_str(offset):
            if offset < 0:
                return f".L{abs(offset)}"
            elif offset > 0:
                return f".R{abs(offset)}"
            elif offset == 0:
                return ""
            else:
                return None

        offset_strs = raw_feat_df_columns.get_level_values(1).map(offset_str)
        return raw_feat_df_columns.get_level_values(0) + offset_strs

    @classmethod
    def _columns_X_to_rdf(cls, X_columns: pd.Index) -> pd.MultiIndex:
        def process_split(l):
            if len(l) == 1:
                offset = 0
            else:
                offset = int(l[2])
                if l[1] == "L":
                    offset = offset * -1
            return (l[0], offset)

        cols_ = [re.split("\.(L|R)([0-9]+)", s) for s in X_columns]
        cols = [process_split(c) for c in cols_]

        return pd.Index(cols)

    @staticmethod
    def _flatten_feat_column_multiindex(feat_df: pd.DataFrame) -> pd.DataFrame:
        def offset_str(offset):
            if offset < 0:
                return f".L{abs(offset)}"
            elif offset > 0:
                return f".R{abs(offset)}"
            elif offset == 0:
                return ""
            else:
                return None

        offset_strs = feat_df.columns.get_level_values(1).map(offset_str)
        new_colnames = feat_df.columns.get_level_values(0) + offset_strs

        feat_df_ = feat_df.copy()
        feat_df_.columns = new_colnames

        return feat_df_

    @classmethod
    def _raw_feat_df_to_X(cls, rdf: pd.DataFrame) -> pd.DataFrame:
        X_cols = cls._columns_rfd_to_X(rdf.columns)
        return rdf.set_axis(X_cols, axis="columns")

    @classmethod
    def _X_to_raw_feat_df(cls, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse of _flatten_feat_column_multiindex"""
        rdf_cols = cls._columns_X_to_rdf(X.columns)
        return X.set_axis(rdf_cols, axis="columns")

    def copy(self, lightweight=False):
        """lightweight: True removes large dataframes from object"""
        c_ = copy.deepcopy(self)
        if lightweight:
            c_._raw_feat_df = None
        return c_

    def dump(self, fn, lightweight=True):
        """lightweight: True removes large dataframes from object"""
        c_ = self
        if lightweight:
            c_ = self.copy(lightweight=True)

        os.makedirs(os.path.dirname(fn), exist_ok=True)
        joblib.dump(c_, fn)

    @classmethod
    def load(cls, fn: str):
        return joblib.load(fn)

    @classmethod
    def load_batch(cls, directory: str, as_list: bool = False):
        assert Path(directory).is_dir()

        fns = glob(f"{directory}/*.ds")
        assert len(fns) > 0

        if as_list:
            return [cls.load(fn) for fn in fns]
        else:
            batch_d = {}
            for fn in fns:
                d = cls.load(fn)
                batch_d[d.name] = d
            return batch_d

    ## transformed Xs

    @property
    def raw_feat_df_T(self):
        self._raw_feat_df_T = getattr(self, "_raw_feat_df_T", None)
        if self._raw_feat_df_T is None:
            transformer = get_transformer(self.feature_transform_id, kind="raw_feat_df")
            self._raw_feat_df_T = transformer(
                self.raw_feat_df, cell_type=self.cell_type
            )
        return self._raw_feat_df_T

    @property
    def XT(self):
        X_cols = self._columns_rfd_to_X(self.raw_feat_df_T.columns)
        return self.raw_feat_df_T.set_axis(X_cols, axis="columns")

    @property
    def XT_train(self):
        return self.XT.loc[self.train_idx, :]

    @property
    def XT_test(self):
        return self.XT.loc[self.test_idx, :]
