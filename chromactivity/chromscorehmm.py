import gzip
import os
import re

import pandas as pd

from chromactivity import TrackInterval, utils
from chromactivity.utils import globs, logger


def _binarize_bw(bw_fn: str, track_interval: TrackInterval, quantile: float):
    s = track_interval.extract_bw(bigwig_fn=bw_fn)

    threshold = s.quantile(quantile)

    binarized_s = s.mask(~s.isna(), s >= threshold).fillna(2).map(int)
    return binarized_s


def generate_tsv_for_cell_type_and_chromosome(
    track_dir, cell_type, chrom, quantile, out_fn
):
    bw_fns = globs(f"{track_dir}/*.bw")

    bw_fns_for_cell_type = []
    for bw_fn in bw_fns:
        cell_type, expert_name = re.findall(".+\/(.+)\.(.+)\.bw", bw_fn)[0]

        if cell_type == cell_type and expert_name != "ChromScore":
            bw_fns_for_cell_type.append(bw_fn)

    track_interval = TrackInterval(
        chrom,
        0,
        utils.chrom_sizes_hg19_d[chrom],
        feature_resolution=25,
    )

    bin_d = {}
    for bw_fn in bw_fns_for_cell_type:
        cell_type, expert_name = re.findall(".+\/(.+)\.(.+)\.bw", bw_fn)[0]

        bin_d[expert_name] = _binarize_bw(bw_fn, track_interval, quantile)

    # combining individual experts to one table
    bin_df = pd.concat(bin_d, axis=1)

    utils.make_parent_dirs(out_fn)
    with gzip.open(out_fn, "wt") as f:
        print(f"{cell_type}\t{chrom}", file=f)
        bin_df.to_csv(f, sep="\t", index=False, header=True)


def generate_binarized_directory(
    track_dir, out_binarized_dir, cell_types=None, chroms=None, quantile=0.98
):
    if cell_types is None:
        cell_types = utils.roadmap_eids

    if chroms is None:
        chroms = utils.chroms_in_order

    for cell_type in cell_types:
        for chrom in chroms:
            out_fn = f"{out_binarized_dir}/{cell_type}.{chrom}_binarized.tsv.gz"
            logger.info(f"Generating {out_fn=}")
            generate_tsv_for_cell_type_and_chromosome(
                track_dir, cell_type, chrom, quantile, out_fn
            )


def chromhmm_learn_model(
    bin_data_dir,
    num_states,
    out_dir,
    memory_mx="48000M",
    assembly="hg19",
    binsize=25,
    numseq=128,
    maxprocessors=4,
    lowmem=True,
    chromhmm_path="vendored/ChromHMM.jar",
):
    params_ = [f"-b {binsize}"]
    if numseq is not None:
        params_ += [f"-n {numseq} -d -1"]
    if maxprocessors is not None:
        params_ += [f"-p {maxprocessors}"]
    if lowmem:
        params_ += ["-lowmem"]

    params_str = " ".join(params_)

    cmd_0 = f"java -mx{memory_mx} -jar {chromhmm_path}"
    cmd_ = f"{cmd_0} LearnModel {params_str} {bin_data_dir} {out_dir} {num_states} {assembly}"
    logger.info(cmd_)

    os.system(cmd_)
