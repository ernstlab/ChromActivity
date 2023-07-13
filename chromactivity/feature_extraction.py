import os
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import pyBigWig
from loguru import logger
from pybedtools import BedTool

from chromactivity import mappings, utils
from chromactivity.dataset import Dataset


class FeatureExtraction:
    """
    Feature extraction for EPIMAP and ROADMAP datasets.

    Includes signal and chrom_state_18_epimap extraction. EPIMAP dataset
    contains no peaks.
    """

    def __init__(
        self, window_size=1000, feature_resolution=25, roadmap_or_epimap="epimap"
    ):
        self.window_len = window_size
        self.feature_resolution = feature_resolution
        self.roadmap_or_epimap = roadmap_or_epimap

        marks_d = {
            "roadmap": mappings.MarksMapping.roadmap_marks,
        }
        self.marks = marks_d[self.roadmap_or_epimap]

    @staticmethod
    def _check_dataset(dataset):
        # check if dataset contains single base regions
        is_single_base = (
            (dataset.coord_df["end"] - dataset.coord_df["start"]) == 1
        ).all()
        assert is_single_base

    @classmethod
    def _extract_signal_interval(
        cls,
        chrom: str,
        start_coord: int,
        end_coord: int,
        bigwig_fn: str,
        feature_resolution: int = 1,
    ) -> List:
        """Set feature_resolution to 1 to get signal at every base pair."""
        try:
            with pyBigWig.open(bigwig_fn) as bw:
                left_pad = 0
                right_pad = 0

                if start_coord < 0:
                    left_pad = 0 - start_coord
                    start_coord = 0

                chrom_len = utils.chrom_sizes_hg19_d[chrom]
                if end_coord >= chrom_len:
                    right_pad = end_coord - chrom_len
                    end_coord = chrom_len

                val = bw.values(chrom, start_coord, end_coord, numpy=True)

                if left_pad > 0 or right_pad > 0:
                    val = np.pad(
                        val,
                        (left_pad, right_pad),
                        "constant",
                        constant_values=(np.nan, np.nan),
                    )

                # downsample signal based on feature resolution
                val = val[::feature_resolution]
        except RuntimeError as e:
            logger.error("Error in pyBigWig, check console for error details")
            raise e

        return val

    @classmethod
    def extract_signal_from_interval_tuples(
        cls, coord_tuples, bigwig_fn, feature_resolution
    ) -> Dict[tuple, np.ndarray]:
        assert os.path.isfile(bigwig_fn), f"File not found: {bigwig_fn}"

        try:
            with pyBigWig.open(bigwig_fn) as bw:
                arr_d = {}
                for chrom, start_coord, end_coord in coord_tuples:
                    left_pad = 0
                    right_pad = 0

                    start_coord_ = start_coord
                    end_coord_ = end_coord

                    if start_coord < 0:
                        left_pad = 0 - start_coord
                        start_coord = 0

                    chrom_len = utils.chrom_sizes_hg19_d[chrom]
                    if end_coord >= chrom_len:
                        right_pad = end_coord - chrom_len
                        end_coord = chrom_len

                    try:
                        assert (
                            start_coord < end_coord
                        ), f"Invalid interval {chrom}:{start_coord}-{end_coord}"
                        assert chrom in [f"chr{c}" for c in range(23)] + [
                            "chrX"
                        ], f"Non-canonical or invalid chromosome: {chrom}"

                        val = bw.values(chrom, start_coord, end_coord, numpy=True)

                        if left_pad > 0 or right_pad > 0:
                            val = np.pad(
                                val,
                                (left_pad, right_pad),
                                "constant",
                                constant_values=(np.nan, np.nan),
                            )

                    except (RuntimeError, AssertionError):
                        # if something goes wrong in extracting bigwig, log the
                        # error, replace with nan and move on
                        logger.debug(
                            (
                                f"pybigwig error in: {bigwig_fn}, "
                                f"{chrom}:{start_coord}-{end_coord}"
                            )
                        )
                        length = end_coord - start_coord
                        val = np.full(length, np.nan)

                    # downsample signal based on feature resolution
                    val = val[::feature_resolution]
                    arr_d[(chrom, start_coord_, end_coord_)] = val
        except RuntimeError as e:
            logger.error(
                f"Error in pyBigWig, check console for error details: {bigwig_fn}"
            )
            raise e

        return arr_d

    @classmethod
    def extract_signal_window_for_coords(
        cls,
        coord_df: pd.DataFrame,
        bigwig_fn: str,
        window_len: int,
        feature_resolution: int,
    ) -> pd.DataFrame:
        """
        There is some overlap with _extract_signal_interval but it is intentional.
        We don't want to open and close the same file over and over again.
        """

        # check all coords are same length
        coord_len_s = coord_df["end"] - coord_df["start"]
        assert (coord_len_s.iloc[0] == coord_len_s).all()
        coord_len = coord_len_s.iloc[0]

        assert os.path.isfile(bigwig_fn), f"File not found: {bigwig_fn}"

        try:
            with pyBigWig.open(bigwig_fn) as bw:
                arr_list = []
                for rr in coord_df.itertuples():
                    chrom = rr.chrom
                    start_coord = rr.start - window_len
                    end_coord = rr.end + window_len

                    left_pad = 0
                    right_pad = 0

                    if start_coord < 0:
                        left_pad = 0 - start_coord
                        start_coord = 0

                    chrom_len = utils.chrom_sizes_hg19_d[chrom]
                    if end_coord >= chrom_len:
                        right_pad = end_coord - chrom_len
                        end_coord = chrom_len

                    try:
                        assert (
                            start_coord < end_coord
                        ), f"Invalid interval {chrom}:{start_coord}-{end_coord}"
                        assert chrom in [f"chr{c}" for c in range(23)] + [
                            "chrX"
                        ], f"Non-canonical or invalid chromosome: {chrom}"

                        val = bw.values(chrom, start_coord, end_coord, numpy=True)

                        if left_pad > 0 or right_pad > 0:
                            val = np.pad(
                                val,
                                (left_pad, right_pad),
                                "constant",
                                constant_values=(np.nan, np.nan),
                            )

                    except (RuntimeError, AssertionError):
                        # if something goes wrong in extracting bigwig, log the
                        # error, replace with nan and move on
                        logger.debug(
                            (
                                f"pybigwig error in: {bigwig_fn}, "
                                f"{chrom}:{start_coord}-{end_coord}"
                            )
                        )
                        length = coord_len + 2 * window_len  # this is the expected size
                        val = np.full(length, np.nan)

                    # downsample signal based on feature resolution
                    val = val[::feature_resolution]
                    arr_list.append(val)
        except RuntimeError as e:
            logger.error(
                f"Error in pyBigWig, check console for error details: {bigwig_fn}"
            )
            raise e

        # this part is implemented in numpy for 30% speed gain
        vals_arr = np.concatenate(arr_list).reshape((coord_df.shape[0], -1))
        feat_df = pd.DataFrame(data=vals_arr, index=coord_df.index)

        columns_ = [-window_len + c * feature_resolution for c in feat_df.columns]
        feat_df.columns = columns_
        feat_df.index = coord_df.index

        return feat_df

    def extract_signal_window_for_dataset(self, dataset: Dataset, mark: str):
        if self.roadmap_or_epimap == "epimap":
            bigwig_fn = mappings.FileMapping.get_epimap_signal_fn(
                mark=mark, cell_type=dataset.cell_type
            )
        elif self.roadmap_or_epimap == "roadmap":
            bigwig_fn = mappings.FileMapping.get_roadmap_signal_fn(
                mark=mark, cell_type=dataset.cell_type
            )
        else:
            raise ValueError()

        signal_feat_df = self.extract_signal_window_for_coords(
            coord_df=dataset.raw_coord_df,
            bigwig_fn=bigwig_fn,
            window_len=self.window_len,
            feature_resolution=self.feature_resolution,
        )

        return signal_feat_df

    @classmethod
    def extract_peak_for_coords(
        cls, coord_df: pd.DataFrame, peak_annot_fn: str, version="v1"
    ):
        if version == "v1":
            return cls.extract_peak_for_coords_v1(coord_df, peak_annot_fn)
        elif version == "v2":
            return cls.extract_peak_for_coords_v2(coord_df, peak_annot_fn)
        else:
            raise ValueError(version)

    @classmethod
    def extract_peak_for_coords_v2(cls, coord_df: pd.DataFrame, peak_annot_fn: str):
        pass

    @classmethod
    def extract_peak_for_coords_v1(cls, coord_df: pd.DataFrame, peak_annot_fn: str):
        with warnings.catch_warnings():  # suppress py38 warning
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            npk_bed = BedTool(peak_annot_fn)
            dataset_bed = BedTool.from_dataframe(coord_df)

            peaks = (
                dataset_bed.intersect(npk_bed, c=True, sorted=True)
                .to_dataframe()
                .iloc[:, -1]
            )
        peaks.rename(0, inplace=True)
        peaks.index = coord_df.index

        return peaks

    @classmethod
    def extract_peaks_combined_for_coords(
        cls, coord_df: pd.DataFrame, peak_annot_fns: List[str]
    ) -> Dict:
        """Combined fast version."""
        assert isinstance(peak_annot_fns, list), "Should be a list"

        dataset_bed = BedTool.from_dataframe(coord_df)
        peaks_combined = dataset_bed.intersect(
            b=peak_annot_fns,
            C=True,
            names=peak_annot_fns,
            sorted=True,
            wao=True,
        ).to_dataframe()

        mark_df_d = {}
        if len(peak_annot_fns) > 1:
            for k, df in peaks_combined.groupby(peaks_combined.columns[-2]):
                assert df.shape[0] == coord_df.shape[0]
                r = df.iloc[:, -1].map(int)
                r.index = coord_df.index
                mark_df_d[k] = r.rename(0)
        elif len(peak_annot_fns) == 1:
            df = peaks_combined
            k = peak_annot_fns[0]

            assert df.shape[0] == coord_df.shape[0]
            r = df.iloc[:, -1].map(int)
            r.index = coord_df.index
            mark_df_d[k] = r.rename(0)

        return mark_df_d

    def extract_peaks_combined_for_dataset(self, dataset, marks):
        if self.roadmap_or_epimap == "epimap":
            raise ValueError("No peak features in epimap")
        elif self.roadmap_or_epimap == "roadmap":
            peak_fns = [
                mappings.FileMapping.get_roadmap_peak_fn(
                    mark=mark, cell_type=dataset.cell_type, is_sorted=True
                )
                for mark in marks
            ]
        else:
            raise ValueError()

        mark_df_d = self.extract_peaks_combined_for_coords(
            dataset.raw_coord_df, peak_annot_fns=peak_fns
        )
        mark_df_d = {mark: v for mark, v in zip(marks, mark_df_d.values())}
        return mark_df_d

    def extract_peak_for_dataset(self, dataset: Dataset, mark: str):
        if self.roadmap_or_epimap == "epimap":
            raise ValueError("No peak features in epimap")
        elif self.roadmap_or_epimap == "roadmap":
            peak_fn = mappings.FileMapping.get_roadmap_peak_fn(
                mark=mark, cell_type=dataset.cell_type
            )
        else:
            raise ValueError()

        return self.extract_peak_for_coords(dataset.raw_coord_df, peak_annot_fn=peak_fn)

    @classmethod
    def extract_chromstate_for_coords(
        cls, coord_df: pd.DataFrame, chromstate_annot_fn: str
    ):
        with warnings.catch_warnings():  # suppress py38 warning
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            npk_bed = BedTool(chromstate_annot_fn)
            dataset_bed = BedTool.from_dataframe(coord_df)

            chrom_state_s = dataset_bed.intersect(npk_bed, wao=True, sorted=True)

        dz = chrom_state_s.to_dataframe(disable_auto_names=True, header=None)

        # sometimes the locus ends up just at the boundary between chrom state calls
        # dz_ = dz[~dz.duplicated(subset=[0, 1, 2, 3], keep="first")]

        # wait actually i don't think this happens, but it breaks when there are
        # repeated coords
        dz_ = dz

        # depending on the input file, the relevant column changes
        col_i = -2
        sample_chromstate = dz_.iloc[0, col_i]
        if "," in sample_chromstate:  # this is the color column
            col_i = 7

        chrom_state_series = dz_.iloc[:, col_i].rename(
            0
        )  # offset = 0, no window around chromstate

        assert coord_df.shape[0] == chrom_state_series.shape[0], (
            coord_df.shape[0],
            chrom_state_series.shape[0],
        )
        chrom_state_series.index = coord_df.index

        return chrom_state_series

    def extract_chromstate_for_dataset(self, dataset: Dataset, chromstate: str):
        chromstate_annot_fn = mappings.ChromStateMapping.get_chromstate_annot_fn(
            chromstate=chromstate,
            cell_type=dataset.cell_type,
            roadmap_or_epimap=self.roadmap_or_epimap,
        )

        chrom_state_series = self.extract_chromstate_for_coords(
            coord_df=dataset.raw_coord_df, chromstate_annot_fn=chromstate_annot_fn
        )

        return chrom_state_series

    @classmethod
    def extract_chromstate_combined_for_coords(
        cls, coord_df: pd.DataFrame, chromstate_annot_fns: List[str]
    ) -> Dict:
        """Combined fast version."""

        assert isinstance(chromstate_annot_fns, list), "Should be a list"
        assert (coord_df["end"] - coord_df["start"] == 1).all()

        dataset_bed = BedTool.from_dataframe(coord_df)
        peaks_combined = dataset_bed.intersect(
            b=chromstate_annot_fns, wao=True, names=chromstate_annot_fns, sorted=True
        ).to_dataframe()

        cs_df_d = {}

        # this changes depending on whether coord_df has a name column
        if len(chromstate_annot_fns) == 1:
            df_ = peaks_combined.reset_index().iloc[:, -2].rename(0)
            df_.index = coord_df.index
            cs_df_d[chromstate_annot_fns[0]] = df_
            assert coord_df.shape[0] == peaks_combined.shape[0], (
                coord_df.shape[0],
                peaks_combined.shape[0],
            )
        else:
            groupby_col = peaks_combined.columns[coord_df.shape[1]]

            for annot_fn, df in peaks_combined.groupby(groupby_col):
                key_str = f"{annot_fn}"
                df_ = df.reset_index().iloc[:, -2].rename(0)
                df_.index = coord_df.index
                cs_df_d[key_str] = df_

            assert coord_df.shape[0] == df.shape[0], (coord_df.shape[0], df.shape[0])

        return cs_df_d

    def extract_chromstate_combined_for_dataset(self, dataset, marks):
        if self.roadmap_or_epimap == "epimap":
            chromstate_annot_fns = [
                mappings.ChromStateMapping.get_chromstate_annot_fn(
                    chromstate=chrom_state,
                    cell_type=self.cell_type,
                    roadmap_or_epimap="epimap",
                )
                for chrom_state in ["chrom_state_18_epimap"]
            ]
        elif self.roadmap_or_epimap == "roadmap":
            chromstate_annot_fns = [
                mappings.ChromStateMapping.get_chromstate_annot_fn(
                    chromstate=chrom_state,
                    cell_type=self.cell_type,
                    roadmap_or_epimap="roadmap",
                )
                for chrom_state in [
                    "chrom_state_15",
                    "chrom_state_18",
                    "chrom_state_25",
                ]
            ]
        else:
            raise ValueError()

        mark_df_d = self.extract_chromstate_combined_for_coords(
            dataset.raw_coord_df, peak_annot_fns=chromstate_annot_fns
        )
        mark_df_d = {mark: v for mark, v in zip(marks, mark_df_d.values())}
        return mark_df_d

    @classmethod
    def extract_closest_tss_distance_for_coords(
        cls, coord_df: pd.DataFrame, gene_annot_fn
    ):
        dataset_bed = BedTool.from_dataframe(coord_df)

        columns = [
            "ensembl_id",
            "chrom",
            "start",
            "end",
            "strand",
            "gene_type",
            "gene_name",
            "gene_full",
        ]

        gene_info = pd.read_csv(
            gene_annot_fn,
            sep="\t",
            on_bad_lines="skip",
            names=columns,
        )

        # need to convert 1 to chr1, etc.
        gene_info["chrom"] = "chr" + gene_info["chrom"]

        gene_info["tss_base"] = gene_info["start"].where(
            gene_info["strand"] == 1, gene_info["end"]
        )
        gene_info["tss_base_"] = gene_info["tss_base"] + 1
        gene_info = gene_info.query("gene_type == 'protein_coding' & chrom != 'chrY'")
        gene_info = gene_info.sort_values(["chrom", "tss_base"])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            tss_bed = BedTool.from_dataframe(
                gene_info[["chrom", "tss_base", "tss_base_", "ensembl_id"]]
            )

            col_names = [
                "chrom",
                "start",
                "end",
                "id",
                "gene_chrom",
                "gene_start",
                "gene_end",
                "ensembl_id",
                "distance_tss",
            ]

            dist_df = dataset_bed.closest(tss_bed, d=True, t="first").to_dataframe(
                disable_auto_names=True, names=col_names
            )

        tss_dist_series = dist_df.iloc[:, -1].rename(0)
        tss_dist_series.index = coord_df.index

        return tss_dist_series

    def extract_closest_tss_distance_for_dataset(
        self, dataset: Dataset, gene_annot="Ensembl_v65"
    ):
        gene_annot_fn = mappings.GeneMapping.gene_annot_fn_d[gene_annot]

        return self.extract_closest_tss_distance_for_coords(
            coord_df=dataset.raw_coord_df, gene_annot_fn=gene_annot_fn
        )

    def extract_raw_features_for_dataset(
        self, dataset: Dataset, marks="all", chrom_state_models=None
    ):
        """marks can be a list or 'all'."""

        if isinstance(marks, str):
            if marks == "all":
                marks = self.marks
            else:
                raise ValueError(f"Invalid marks {marks}")

        if chrom_state_models is None:
            if self.roadmap_or_epimap == "roadmap":
                chrom_state_models = [
                    "chrom_state_15",
                    "chrom_state_18",
                    "chrom_state_25",
                ]
            elif self.roadmap_or_epimap == "epimap":
                chrom_state_models = ["chrom_state_18_epimap"]
            else:
                raise ValueError()

        mark_df_d = {}

        # extract signal features
        for mark in marks:
            signal_df = self.extract_signal_window_for_dataset(
                dataset=dataset, mark=mark
            )
            key_str = f"feat_signal_{mark}"
            mark_df_d[key_str] = signal_df

        # extract peak features if exists

        if self.roadmap_or_epimap == "roadmap":
            logger.info("Running new version")
            d_ = self.extract_peaks_combined_for_dataset(dataset=dataset, marks=marks)
            d_ = {f"feat_peak_{mark}": v for mark, v in d_.items()}
            mark_df_d = {**mark_df_d, **d_}

        # extract chrom state features
        for chrom_state_model in chrom_state_models:
            try:
                chrom_state_s = self.extract_chromstate_for_dataset(
                    dataset=dataset, chromstate=chrom_state_model
                )

                key_str = f"feat_{chrom_state_model}"
                mark_df_d[key_str] = chrom_state_s
            except FileNotFoundError as e:
                logger.error(f"Chromatin state extraction failed: {chrom_state_model}")
                logger.exception(e)
            except Exception as e:
                logger.error(f"Chromatin state extraction failed: {chrom_state_model}")
                raise e

        key_str = "feat_dist_tss"
        mark_df_d[key_str] = pd.Series(dtype="object")

        raw_feat_df = pd.concat(mark_df_d, axis=1)
        return raw_feat_df

    def dump_raw_feature_table(self, dataset: Dataset, out_fn: str):
        Path(out_fn).parent.mkdir(parents=True, exist_ok=True)
        raw_feat_df = self.extract_raw_features_for_dataset(dataset, marks="all")
        joblib.dump(raw_feat_df, out_fn)
        return raw_feat_df
