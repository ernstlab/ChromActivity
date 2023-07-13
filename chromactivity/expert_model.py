from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from chromactivity import Dataset, mappings, utils
from chromactivity.utils import logger


@dataclass
class ExpertModel:
    train_dataset: Dataset = field(repr=False)
    pipeline: Pipeline = field(default=None, repr=False)
    name: str = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"model_{self.train_dataset.name}"

        if self.pipeline is None:
            self.pipeline = self.get_pipeline(
                version="2022-10-31", train_dataset=self.train_dataset
            )

    def fit(self):
        ds = self.train_dataset

        if ds._train_idx is None:
            logger.warning(f"{self.name}: Train/test not set, will train everywhere")

            ds._train_idx = pd.Series(True, ds.bed_df.index)
            ds._test_idx = pd.Series(False, ds.bed_df.index)

        assert self.pipeline is not None, "pipeline is None"
        assert ds.sample_weights is None, "sample_weights not supported"

        self.pipeline.fit(
            X=ds.XT_train, y=ds.y_train.map({"activating": 1, "neutral": 0})
        )

        return self

    def infer(self, X):
        return pd.DataFrame(
            self.pipeline.predict_proba(X),
            index=X.index,
            columns=self.pipeline.classes_,
        )

    def infer_activating(self, X):
        return self.infer(X)[1]

    def dump(self, fn):
        # clear training datasets raw feature matrix before dumping model to
        # avoid generating huge models
        _raw_feat_df_ = self.train_dataset._raw_feat_df
        self.train_dataset._raw_feat_df = None

        # create enclosing directory
        Path(fn).parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(self, fn)

        # put it back
        self.train_dataset._raw_feat_df = _raw_feat_df_

    @classmethod
    def get_pipeline(cls, version="2022-10-31", train_dataset=None):
        if version == "2022-10-31":

            def make_column_transformer(
                signal_marks=mappings.MarksMapping.roadmap_marks,
            ) -> ColumnTransformer:
                transformers = []

                for feature_name in signal_marks:
                    current_feat_signal = f"feat_signal_{feature_name}"
                    signal_pca_transformer = (
                        f"pca_{current_feat_signal}",
                        PCA(n_components=3, whiten=False),
                        make_column_selector(f"{current_feat_signal}_*"),
                    )

                    transformers.append(signal_pca_transformer)

                ct = ColumnTransformer(
                    transformers=transformers,
                    remainder="passthrough",
                    verbose_feature_names_out=True,
                )

                return ct

            def get_class_weight(dataset, f=3.0):
                vc = dataset.labels.value_counts().to_dict()
                activating_weight = (vc["neutral"] / vc["activating"]) / f

                if np.isclose(activating_weight, 1.0):
                    return None
                else:
                    class_weight = {
                        1: activating_weight,
                        0: 1,
                    }
                    return class_weight

            if train_dataset is None:
                logger.warning("train_dataset not set, class_weight will be None")
                class_weight = None
            else:
                class_weight = get_class_weight(train_dataset, f=3.0)

            model = LogisticRegression(
                n_jobs=8,
                random_state=1,
                max_iter=15000,
                C=1,
                penalty="l2",
                class_weight=class_weight,
            )

            bmodel = BaggingClassifier(
                base_estimator=model,
                n_estimators=100,
                random_state=1,
                n_jobs=4,
                # warm_start=True,
                # oob_score=True,
                # max_samples=0.9,
            )

            column_transformer = make_column_transformer()

            p = make_pipeline(column_transformer, StandardScaler(), bmodel)
            return p
        else:
            raise ValueError(f"Invalid pipeline version: {version}")

    @classmethod
    def load(cls, fn):
        return utils.load(fn)

    @classmethod
    def get_trained_expert(cls, tsv_fn: str):
        """ """
        eid_mapping = {
            "a549": "E114",
            "gm": "E116",
            "helaS3": "E117",
            "hepg2": "E118",
            "k562": "E123",
        }

        ds_name = Path(tsv_fn).stem

        cell_type = ds_name.split("-")[1]
        eid = eid_mapping[cell_type]

        df = pd.read_csv(tsv_fn, sep="\t")
        ds = Dataset(
            name=ds_name,
            cell_type=eid,
            bed_df=df,
            feature_transform_id="ft_20220505_cs25",
            test_chromosomes=[],
        )

        ds.extract_features(dump=False)

        pipeline = cls.get_pipeline(train_dataset=ds)
        em_ = cls(name=ds_name, train_dataset=ds, pipeline=pipeline)

        em_.fit()
        return em_
