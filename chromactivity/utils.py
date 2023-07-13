import copy
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm

from chromactivity import mappings


def get_logger(level="INFO"):
    from loguru import logger

    try:
        logger.remove(handler_id=0)
        logger.add(sink=sys.stdout, level=level, colorize=True)
    except ValueError:
        pass

    return logger


logger = get_logger()


def set_notebook_options():
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    plt.matplotlib.rcParams["figure.dpi"] = 210
    set_matplotlib_formats("retina")

    pd.set_option("display.max_columns", 100)


def globs(fn, **kwargs):
    from glob import glob

    return sorted(glob(fn, **kwargs))


def split_chunks(l, n):  # noqa: E741
    return [l[i : i + n] for i in range(0, len(l), n)]


def dump(obj, fn, **kwargs):
    """Dump object with joblib, also make parent directories."""

    p = Path(fn).parent
    p.mkdir(exist_ok=True, parents=True)

    joblib.dump(obj, fn, **kwargs)
    return fn


def load(fn, **kwargs):
    try:
        obj = joblib.load(fn, **kwargs)
    except Exception as e:
        logger.error(e)
        return None
    return obj


def dump_parquet(df: pd.DataFrame, fn: str, **kwargs):
    """Dump DataFrame as parquet, also make parent directories."""

    p = Path(fn).parent
    p.mkdir(exist_ok=True, parents=True)

    df.to_parquet(path=fn, **kwargs)
    return fn


def deepcopy(obj):
    return copy.deepcopy(obj)


def load_glob(fn_glob, **kwargs):
    """Returned in lex sort order."""
    fns = globs(fn_glob, **kwargs)
    if len(fns) < 1:
        raise ValueError(f"Glob returned no results: {fn_glob}")

    l = []
    for fn in tqdm(fns):
        l.append(load(fn))

    return l


def dump_as_json(obj, fn, **kwargs):
    with open(fn, "w") as f:
        json.dump(obj, f, indent=2, **kwargs)
    return fn


def make_parent_dirs(fn):
    """If fn ends with a /, create as directory."""
    if fn:
        if fn[-1] == "/":
            p = Path(fn)
        else:
            p = Path(fn).parent

        p.mkdir(exist_ok=True, parents=True)


def exists(fn):
    return Path(fn).expanduser().exists()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d.%Hh%M")


def read_bed(bed_fn):
    df = pd.read_csv(bed_fn, sep="\t", header=None)
    if df.shape[1] == 4:
        col_names = ["chrom", "start", "end", "name"]
    elif df.shape[1] == 5:
        col_names = ["chrom", "start", "end", "name", "value"]
    else:
        raise ValueError("Bed file must contain 4 or 5 columns.")
    df.columns = col_names
    return df


_c = """
chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,
chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX
"""
chroms_in_order = [s.strip() for s in _c.split(",")]

chrom_sizes_hg19_d = {
    "chr1": 249250621,
    "chr2": 243199373,
    "chr3": 198022430,
    "chr4": 191154276,
    "chr5": 180915260,
    "chr6": 171115067,
    "chr7": 159138663,
    "chrX": 155270560,
    "chr8": 146364022,
    "chr9": 141213431,
    "chr10": 135534747,
    "chr11": 135006516,
    "chr12": 133851895,
    "chr13": 115169878,
    "chr14": 107349540,
    "chr15": 102531392,
    "chr16": 90354753,
    "chr17": 81195210,
    "chr18": 78077248,
    "chr20": 63025520,
    "chr19": 59128983,
    "chr22": 51304566,
    "chr21": 48129895,
}

_e = """
E017,E002,E008,E001,E015,E014,E016,E003,E024,E020,E019,E018,E021,E022,E007,E009,
E010,E013,E012,E011,E004,E005,E006,E062,E034,E045,E033,E044,E043,E039,E041,E042,
E040,E037,E048,E038,E047,E029,E031,E035,E051,E050,E036,E032,E046,E030,E026,E049,
E025,E023,E052,E055,E056,E059,E061,E057,E058,E028,E027,E054,E053,E112,E093,E071,
E074,E068,E069,E072,E067,E073,E070,E082,E081,E063,E100,E108,E107,E089,E090,E083,
E104,E095,E105,E065,E078,E076,E103,E111,E092,E085,E084,E109,E106,E075,E101,E102,
E110,E077,E079,E094,E099,E086,E088,E097,E087,E080,E091,E066,E098,E096,E113,E114,
E115,E116,E117,E118,E119,E120,E121,E122,E123,E124,E125,E126,E127,E128,E129
"""

roadmap_eids = [s.strip() for s in _e.split(",")]

_e = """
E003,E004,E005,E006,E007,E011,E012,E013,E016,E024,E027,E028,E037,E038,
E047,E050,E053,E054,E055,E056,E057,E058,E059,E061,E062,E065,E066,E070,
E071,E079,E082,E084,E085,E087,E094,E095,E096,E097,E098,E100,E104,E105,
E106,E109,E112,E113,E114,E116,E117,E118,E119,E120,E122,E123,E127,E128
"""

roadmap_expr_eids = [s.strip() for s in _e.split(",")]


def merge_bigwigs(bw_fns, bw_out_fn):
    """merge bigwigs by converting to wig first"""

    p = Path(bw_out_fn).parent
    p.mkdir(exist_ok=True, parents=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, bw_fn in enumerate(bw_fns):
            wig_out_fn = f"{tmpdir}/{i:04n}.wig"
            cmd_ = f"{mappings.paths['bigWigToWig']} {bw_fn} {wig_out_fn}"
            logger.info(cmd_)
            subprocess.run(cmd_, shell=True)

        wig_fns = " ".join(globs(f"{tmpdir}/*.wig"))
        wig_out_fn = f"{tmpdir}/combined.w"

        cmd_ = f"cat {wig_fns} > {wig_out_fn}"
        logger.info(cmd_)
        subprocess.run(cmd_, shell=True)

        chrom_sizes_fn = mappings.paths["chromsizes_hg19"]
        cmd_ = (
            f"{mappings.paths['wigToBigWig']} {wig_out_fn} {chrom_sizes_fn} {bw_out_fn}"
        )
        logger.info(cmd_)
        subprocess.run(cmd_, shell=True)

    if Path(bw_out_fn).exists():
        logger.info(f"Merged: {bw_out_fn}")
    else:
        logger.error(f"Error generating: {bw_out_fn}")
