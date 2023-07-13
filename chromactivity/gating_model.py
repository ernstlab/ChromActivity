from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from chromactivity import ExpertModel, TrackInterval, mappings, utils

logger = utils.logger


@dataclass
class GatingModel:
    gating_pipeline: Pipeline = field(default=None, repr=False)
    expert_models_d: dict[str, ExpertModel] = field(default=None, repr=False)
    name: str = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"gating_{utils.now_str()}"

        if self.gating_pipeline is None:
            self.gating_pipeline = self.get_gating_pipeline(version="2022-10-31")

    def fit_gating(self, sample_n=2000):
        datasets = [em.train_dataset for em in self.expert_models_d.values()]
        self.gating_pipeline = self._fit_gating_datasets(
            datasets, self.gating_pipeline, sample_n=sample_n
        )

    @classmethod
    def _fit_gating_datasets(cls, datasets, gating_pipeline, sample_n=2000):
        X_l = []
        targets = []
        for ds in datasets:
            X_s = ds.XT_train.sample(n=sample_n, replace=True, random_state=1)
            X_l.append(X_s)
            targets.append(pd.Series(ds.name, index=X_s.index))

        X_ = pd.concat(X_l, ignore_index=True)
        y_ = pd.concat(targets, ignore_index=True)

        gating_pipeline.fit(X_, y_)
        return gating_pipeline

    @classmethod
    def get_gating_pipeline(cls, version="2022-10-31"):
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

            model = LogisticRegression(
                n_jobs=8,
                random_state=1,
                max_iter=15000,
                C=1,
                penalty="l2",
                class_weight=None,
            )

            column_transformer = make_column_transformer()

            p = make_pipeline(column_transformer, StandardScaler(), model)
            return p
        else:
            raise ValueError(f"Invalid pipeline version: {version}")

    def _get_weights_df(self, X: pd.DataFrame):
        weights_df = pd.DataFrame(
            self.gating_pipeline.predict_proba(X),
            X.index,
            columns=self.gating_pipeline.classes_,
        )
        return weights_df

    def _get_scores_df(self, X: pd.DataFrame):
        scores_d = {}
        for ds_name, model in self.expert_models_d.items():
            scores_d[ds_name] = model.infer_activating(X)

        scores_df = pd.concat(scores_d, axis=1)
        return scores_df

    def _get_combined_scores(self, X: pd.DataFrame):
        wdf = self._get_weights_df(X)
        sdf = self._get_scores_df(X)

        try:
            gated_scores = (wdf * sdf.loc[:, wdf.columns]).sum(axis=1)
        except KeyError as e:
            logger.info("Gating model might be incompatible")
            logger.info(f"{e}")

            wdf_ = wdf.loc[:, sdf.columns]
            wdf_renorm = wdf_.div(wdf_.sum(axis=1), axis=0)

            gated_scores = (wdf_renorm * sdf).sum(axis=1)

        mean_scores = sdf.mean(axis=1)

        return [gated_scores, mean_scores, wdf, sdf]

    def _get_combined_scores_means_only(self, X: pd.DataFrame):
        sdf = self._get_scores_df(X)
        mean_scores = sdf.mean(axis=1)

        return [None, mean_scores, None, sdf]

    def infer(self, X: pd.DataFrame):
        expert_scores_df = self._get_scores_df(X)
        chromscore = expert_scores_df.mean(axis=1).rename("ChromScore")

        return chromscore, expert_scores_df

    def dump(self, fn, lightweight=True):
        if lightweight:
            # clear training datasets raw feature matrix before dumping model to
            # avoid generating huge models
            dataset_d = {}
            for em_name, em in self.expert_models_d.items():
                dataset_d[em_name] = em.train_dataset
                em.train_dataset = None

            utils.dump(self, fn)

            # put them back
            for em_name, em in self.expert_models_d.items():
                em.train_dataset = dataset_d[em_name]
        else:
            utils.dump(self, fn)

    @classmethod
    def get_trained_chromscore_model(cls, labels_dir="data/labels"):
        tsv_fns = utils.globs(f"{labels_dir}/*.tsv")

        em_d = {}
        for tsv_fn in tsv_fns:
            logger.info(f"{tsv_fn=}")
            em_ = ExpertModel.get_trained_expert(tsv_fn)
            em_d[em_.name] = em_

        gm_ = GatingModel(name="chromscore", expert_models_d=em_d)
        return gm_

    @classmethod
    def generate_tracks_from_model(
        cls,
        gm,
        cell_type,
        coords_tuples,
        combined_bigwigs_out_dir,
        tmp_dir_path,
        part_length=1_000_000,
    ):
        bigwigs_out_dir = f"{tmp_dir_path}/{combined_bigwigs_out_dir}"

        ti_l = [
            TrackInterval(
                chrom,
                start_pos,
                end_pos,
                cell_type=cell_type,
                feature_transform_id="ft_20220505_cs25",
            )
            for (chrom, start_pos, end_pos) in coords_tuples
        ]

        logger.info(f"{bigwigs_out_dir=}")

        # Generate intermediate bigwigs
        for ti in ti_l:
            ti.generate_gating_model_bigwigs_all(
                gating_model=gm,
                gating_bigwigs_out_dir=bigwigs_out_dir,
                part_length=part_length,
                tmp_dir_path=tmp_dir_path,
            )

        # Merging to final bigwigs

        # ChromScore tracks
        bw_fns = utils.globs(f"{bigwigs_out_dir}/mean/{cell_type}.*/*")
        utils.merge_bigwigs(
            bw_fns, bw_out_fn=f"{combined_bigwigs_out_dir}/{cell_type}.ChromScore.bw"
        )

        # Expert tracks
        expert_dataset_names = sorted(
            set(
                Path(fn).stem
                for fn in utils.globs(f"{bigwigs_out_dir}/individual/{cell_type}.*/*")
            )
        )

        for expert_dataset_name in tqdm(expert_dataset_names):
            bw_fns = utils.globs(
                f"{bigwigs_out_dir}/individual/{cell_type}.*/{expert_dataset_name}.bw"
            )
            utils.merge_bigwigs(
                bw_fns,
                bw_out_fn=f"{combined_bigwigs_out_dir}/{cell_type}.{expert_dataset_name}.bw",
            )
