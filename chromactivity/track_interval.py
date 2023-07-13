import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import pybedtools
from loguru import logger
from tqdm import tqdm

from chromactivity import mappings
from chromactivity.dataset import Dataset
from chromactivity.feature_extraction import FeatureExtraction


@dataclass
class TrackInterval:
    chrom: str
    start: int
    end: int

    cell_type: str = None

    window_size: int = 1000
    feature_resolution: int = 25
    roadmap_or_epimap: str = "roadmap"

    feature_transform_id: str = None  # if this is set, we do transform

    _raw_feat_df: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)
    _raw_feat_df_T: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.name = (
            f"TrackInterval[{self.chrom}:{self.start}-{self.end}][{self.cell_type}]"
        )
        self._raw_feat_df = None

        marks_d = {
            "roadmap": mappings.MarksMapping.roadmap_marks,
        }
        self.marks = marks_d[self.roadmap_or_epimap]

    @property
    def name_safe(self):
        return f"{self.cell_type}.{self.chrom}.{self.start}-{self.end}"

    @property
    def coord_df(self):
        coords_ = {
            "start": pd.Series(range(self.start, self.end, self.feature_resolution))
        }
        coords_["end"] = coords_["start"] + 1

        coords_df = pd.DataFrame(coords_)
        coords_df["chrom"] = self.chrom
        coords_df["region_id"] = "."

        coords_df = coords_df[["chrom", "start", "end", "region_id"]]
        return coords_df

    def extract_signal_window_interval(self, mark, version="v2") -> pd.DataFrame:
        if version == "v1":
            return self.extract_signal_window_interval_cdf_based(mark)
        elif version == "v2":
            return self.extract_signal_window_interval_interval_optimized(mark)

    def extract_signal_window_interval_interval_optimized(self, mark) -> pd.DataFrame:
        if self.roadmap_or_epimap == "roadmap":
            bigwig_fn = mappings.FileMapping.get_roadmap_signal_fn(
                mark=mark, cell_type=self.cell_type
            )
        else:
            raise ValueError()

        coord_tuple = (
            self.chrom,
            self.start - self.window_size,
            self.end + self.window_size,
        )

        signal_d = FeatureExtraction.extract_signal_from_interval_tuples(
            coord_tuples=[coord_tuple],
            bigwig_fn=bigwig_fn,
            feature_resolution=self.feature_resolution,
        )

        total_window_index_len = self.window_size * 2 // self.feature_resolution + 1

        arr = signal_d[coord_tuple]
        arr_l = []
        for start_ in range(0, arr.shape[0] - total_window_index_len + 1):
            end_ = start_ + total_window_index_len
            arr_l.append(arr[start_:end_])

        vals_arr = np.concatenate(
            arr_l,
        ).reshape((-1, total_window_index_len))
        signal_feat_df = pd.DataFrame(
            data=vals_arr,
            columns=pd.RangeIndex(
                start=-self.window_size,
                stop=self.window_size + 1,
                step=self.feature_resolution,
            ),
        )
        return signal_feat_df

    def extract_signal_window_interval_cdf_based(self, mark) -> pd.DataFrame:
        if self.roadmap_or_epimap == "roadmap":
            bigwig_fn = mappings.FileMapping.get_roadmap_signal_fn(
                mark=mark, cell_type=self.cell_type
            )
        else:
            raise ValueError()

        signal_feat_df = FeatureExtraction.extract_signal_window_for_coords(
            self.coord_df, bigwig_fn, self.window_size, self.feature_resolution
        )
        return signal_feat_df

    def extract_chrom_state_interval(self, chrom_state_model) -> pd.Series:
        chromstate_annot_fn = mappings.ChromStateMapping.get_chromstate_annot_fn(
            chromstate=chrom_state_model,
            cell_type=self.cell_type,
            roadmap_or_epimap=self.roadmap_or_epimap,
        )

        return FeatureExtraction.extract_chromstate_for_coords(
            coord_df=self.coord_df,
            chromstate_annot_fn=chromstate_annot_fn,
        )

    def extract_chrom_state_combined_for_interval(
        self, chrom_state_models
    ) -> Dict[str, pd.DataFrame]:
        chromstate_annot_fns = [
            mappings.ChromStateMapping.get_chromstate_annot_fn(
                chromstate=chrom_state,
                cell_type=self.cell_type,
                roadmap_or_epimap=self.roadmap_or_epimap,
            )
            for chrom_state in chrom_state_models
        ]

        cs_df_d = FeatureExtraction.extract_chromstate_combined_for_coords(
            coord_df=self.coord_df,
            chromstate_annot_fns=chromstate_annot_fns,
        )

        return {
            cs_model: v for cs_model, v in zip(chrom_state_models, cs_df_d.values())
        }

    def extract_peak_for_interval(self, mark) -> pd.Series:
        peak_annot_fn = mappings.FileMapping.get_roadmap_peak_fn(
            mark=mark, cell_type=self.cell_type
        )

        return FeatureExtraction.extract_peak_for_coords(
            self.coord_df, peak_annot_fn=peak_annot_fn
        )

    def extract_peaks_combined_for_interval(self, marks):
        peak_fns = [
            mappings.FileMapping.get_roadmap_peak_fn(
                mark=mark, cell_type=self.cell_type, is_sorted=True
            )
            for mark in marks
        ]

        mark_df_d = FeatureExtraction.extract_peaks_combined_for_coords(
            self.coord_df, peak_fns
        )
        mark_df_d = {mark: v for mark, v in zip(marks, mark_df_d.values())}
        return mark_df_d

    def extract_raw_features(self, version="v1"):
        logger.debug(f"Computing features for {self.name}")
        marks = self.marks

        mark_df_d = {}

        # signal features
        for mark in marks:
            signal_df = self.extract_signal_window_interval(mark=mark)

            key_str = f"feat_signal_{mark}"
            mark_df_d[key_str] = signal_df

        if self.roadmap_or_epimap == "roadmap":
            chrom_state_models = ["chrom_state_25"]
        else:
            raise ValueError()

        if version == "v1":
            # extract peak features if exists
            if self.roadmap_or_epimap == "roadmap":
                for mark in marks:
                    peak_s = self.extract_peak_for_interval(mark=mark)
                    key_str = f"feat_peak_{mark}"
                    mark_df_d[key_str] = peak_s
        elif version == "v2":
            if self.roadmap_or_epimap == "roadmap":
                logger.info("Running new version")
                d_ = self.extract_peaks_combined_for_interval(marks=marks)

                d_ = {f"feat_peak_{mark}": v for mark, v in d_.items()}
                mark_df_d = {**mark_df_d, **d_}

        # chrom state features
        for chrom_state_model in chrom_state_models:
            try:
                chrom_state_s = self.extract_chrom_state_interval(
                    chrom_state_model=chrom_state_model,
                )
            except Exception as e:
                logger.info(f"Chrom state extraction failed, may not exist: {e}")
                chrom_state_s = pd.Series(np.nan, index=self.coord_df.index)

            key_str = f"feat_{chrom_state_model}"
            mark_df_d[key_str] = chrom_state_s

        key_str = "feat_dist_tss"
        mark_df_d[key_str] = pd.Series(dtype="object")

        pybedtools.cleanup()

        return pd.concat(mark_df_d, axis=1)

    @property
    def raw_feat_df(self):
        if self._raw_feat_df is None:
            self._raw_feat_df = self.extract_raw_features()
        return self._raw_feat_df

    @property
    def raw_feat_df_T(self):
        if not self.feature_transform_id:
            raise ValueError()
        else:
            if self._raw_feat_df_T is None:
                from chromactivity.ft_mapping import get_transformer

                fun = get_transformer(self.feature_transform_id, kind="raw_feat_df")
                logger.debug(f"Transforming: {self.feature_transform_id}")
                self._raw_feat_df_T = fun(self.raw_feat_df, cell_type=self.cell_type)
            return self._raw_feat_df_T

    @property
    def X(self):
        return Dataset._flatten_feat_column_multiindex(self.raw_feat_df)

    @property
    def XT(self):
        return Dataset._flatten_feat_column_multiindex(self.raw_feat_df_T)

    def dump(self, fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        joblib.dump(self, fn)

    @classmethod
    def load(cls, fn):
        return joblib.load(fn)

    @classmethod
    def expand_interval(cls, interval_str):
        # e.g. "chr8:28,671,427-29,137,319"
        chrom, start, end = interval_str.replace(",", "").replace(":", "-").split("-")
        return chrom, int(start), int(end)

    def generate_wig_from_score_series(self, score, out_fn, skip_header=False):
        cdf = self.coord_df.copy()
        cdf["start"] = cdf["start"] - self.feature_resolution // 2
        cdf["end"] = cdf["end"] + self.feature_resolution // 2

        assert cdf.shape[0] == score.shape[0]
        cdf["score"] = score

        chrom = cdf.iloc[0, 0]
        start = cdf.iloc[0, 1]

        wig_str = (
            f"""fixedStep chrom={chrom} start={start} step={self.feature_resolution}"""
        )
        scores_str = [f"{s:.4f}" for s in score.values.tolist()]

        if skip_header:
            wig_str = "\n".join(scores_str)
        else:
            wig_str = "\n".join([wig_str] + scores_str)

        with open(out_fn, "w") as f:
            print(wig_str, file=f)

    def track_interval_generator(self, part_length):
        for start_ in range(self.start, self.end, part_length):
            end_ = start_ + part_length
            if end_ > self.end:
                end_ = self.end

            ti = TrackInterval(
                chrom=self.chrom,
                start=start_,
                end=end_,
                cell_type=self.cell_type,
                feature_resolution=self.feature_resolution,
                roadmap_or_epimap=self.roadmap_or_epimap,
                window_size=self.window_size,
                feature_transform_id=self.feature_transform_id,
            )

            ti.start = start_
            ti.end = end_

            yield ti

    @classmethod
    def combine_wigs_to_bigwig(cls, ordered_fns, out_bw):
        wigToBigWig_path = mappings.paths["wigToBigWig"]
        chromsizes_path = mappings.paths["chromsizes_hg19"]

        with tempfile.TemporaryDirectory() as tmpdir:
            wig_out_fn = f"{tmpdir}/out.combined.wig"
            with open(wig_out_fn, "w") as f:
                for current_fn in tqdm(ordered_fns, desc="concatenating"):
                    with open(current_fn) as ff:
                        f.write(ff.read())

            cmd_ = f"{wigToBigWig_path} {wig_out_fn} {chromsizes_path} {out_bw}"
            logger.info(cmd_)
            _ = subprocess.run(cmd_, text=True, shell=True)

    # gating model bigwig outputs
    def _generate_multipart_wigs_gating_model(
        self, gating_model, part_length, out_directory
    ):
        for i, ti_ in enumerate(self.track_interval_generator(part_length)):
            logger.info(f"Running for: {ti_.name}")

            (
                gated_scores,
                mean_scores,
                weights_df,
                scores_df,
            ) = gating_model._get_combined_scores_means_only(
                ti_.XT,
            )

            if gated_scores is not None:
                # write gated scores
                out_fn = f"{out_directory}/gated/{i:08d}.{ti_.chrom}_{ti_.start}_{ti_.end}.wig"
                Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
                ti_.generate_wig_from_score_series(
                    gated_scores, out_fn, skip_header=False
                )
                logger.debug(f"Wrote as wig: {out_fn}")

            if mean_scores is not None:
                # write gated scores
                out_fn = f"{out_directory}/mean/{i:08d}.{ti_.chrom}_{ti_.start}_{ti_.end}.wig"
                Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
                ti_.generate_wig_from_score_series(
                    mean_scores, out_fn, skip_header=False
                )
                logger.debug(f"Wrote as wig: {out_fn}")

            if scores_df is not None:
                # write individual model scores
                for col in scores_df.columns:
                    out_fn = f"{out_directory}/individual/{col}/{i:08d}.{ti_.chrom}_{ti_.start}_{ti_.end}.wig"
                    s_ = scores_df[col]
                    Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
                    ti_.generate_wig_from_score_series(s_, out_fn, skip_header=False)
                    logger.debug(f"Wrote as wig: {out_fn}")

            if weights_df is not None:
                # write gating df scores
                for col in weights_df.columns:
                    out_fn = f"{out_directory}/gating_coefs/{col}/{i:08d}.{ti_.chrom}_{ti_.start}_{ti_.end}.wig"
                    s_ = weights_df[col]
                    Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
                    ti_.generate_wig_from_score_series(s_, out_fn, skip_header=False)
                    logger.debug(f"Wrote as wig: {out_fn}")

    def _do_bigwigs(self, multipart_wig_directory, gating_bigwigs_out_dir):
        out_directory = multipart_wig_directory

        def combine_wig_to_bigwigs_multi(out_dir_wigs, path_bigwig_out_dir):
            fns_l = [
                sorted(glob(f"{fnn}/*")) for fnn in sorted(glob(f"{out_dir_wigs}/*"))
            ]

            logger.info(f"Writing bigwigs to {path_bigwig_out_dir}")
            for fns in fns_l:
                out_fn = (
                    f"{path_bigwig_out_dir}/{self.name_safe}/"
                    + fns[0].split("/")[-2]
                    + ".bw"
                )
                Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
                TrackInterval.combine_wigs_to_bigwig(fns, out_fn)

        # ChromScore
        out_dir_wigs = f"{out_directory}/mean/"
        path_bigwig_out_dir = f"{gating_bigwigs_out_dir}/mean"
        fns = sorted(glob(f"{out_dir_wigs}/*"))

        if fns:
            logger.info(f"Writing bigwigs to {path_bigwig_out_dir}")
            out_fn = (
                f"{path_bigwig_out_dir}/{self.name_safe}/"
                + fns[0].split("/")[-2]
                + ".bw"
            )
            Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
            TrackInterval.combine_wigs_to_bigwig(fns, out_fn)

        # individual scores
        out_dir_wigs = f"{out_directory}/individual/"
        path_bigwig_out_dir = f"{gating_bigwigs_out_dir}/individual"
        if glob(f"{out_dir_wigs}/*"):  # if any files exist
            combine_wig_to_bigwigs_multi(out_dir_wigs, path_bigwig_out_dir)

    def generate_gating_model_bigwigs_all(
        self,
        gating_model,
        gating_bigwigs_out_dir,
        part_length=1_000_000,
        tmp_dir_path=None,
    ):
        with tempfile.TemporaryDirectory(dir=tmp_dir_path) as tmp_dir:
            self._generate_multipart_wigs_gating_model(
                gating_model, part_length, tmp_dir
            )
            self._do_bigwigs(tmp_dir, gating_bigwigs_out_dir)

    def extract_bw(self, bigwig_fn):
        coord_tuples = [(self.chrom, self.start, self.end)]

        s = FeatureExtraction.extract_signal_from_interval_tuples(
            bigwig_fn=bigwig_fn,
            coord_tuples=coord_tuples,
            feature_resolution=self.feature_resolution,
        )[coord_tuples[0]]

        return pd.Series(s, index=self.coord_df.index)

    @classmethod
    def bw_average(cls, bw_fns, chroms, out_fn, chrom_mean_dir):
        from chromactivity import utils

        utils.make_parent_dirs(chrom_mean_dir)

        for current_chrom in tqdm(chroms):
            logger.info(current_chrom)
            ti = TrackInterval(
                current_chrom,
                start=10_000,
                end=utils.chrom_sizes_hg19_d[current_chrom] - 10_000,
                cell_type="E000",
            )

            signal_d = {}
            for bigwig_fn in bw_fns:
                signal_d[bigwig_fn] = ti.extract_bw(bigwig_fn=bigwig_fn)

            df = pd.concat(signal_d, axis=1)
            mean_score = df.mean(axis=1)

            with tempfile.TemporaryDirectory() as wig_dir:
                ti.generate_wig_from_score_series(
                    score=mean_score, out_fn=f"{wig_dir}/{ti.name_safe}.wig"
                )
                fns_ = utils.globs(f"{wig_dir}/*.wig")
                ti.combine_wigs_to_bigwig(
                    ordered_fns=fns_, out_bw=f"{chrom_mean_dir}/{current_chrom}.mean.bw"
                )

        utils.merge_bigwigs(
            bw_fns=utils.globs(f"{chrom_mean_dir}/*.mean.bw"), bw_out_fn=out_fn
        )
        logger.info(f"Done: {out_fn}")
