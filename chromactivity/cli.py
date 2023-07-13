import argparse
import sys
import tempfile


def train_experts(labels_dir="data/labels", model_out_fn="models/chromactivity.model"):
    from chromactivity import GatingModel

    model = GatingModel.get_trained_chromscore_model(labels_dir=labels_dir)
    model.dump(model_out_fn, lightweight=True)


def generate_tracks(
    model_fn="models/chromactivity.model",
    cell_types=None,
    combined_bigwigs_out_dir="tracks",
    coords_bed_fn=None,
    tmp_dir_path=None,
    part_length=1_000_000,
):
    import pandas as pd
    from chromactivity import GatingModel, utils

    logger = utils.logger

    # Load model
    gm = utils.load(model_fn)

    # All cell types
    if cell_types is None:
        cell_types = utils.roadmap_eids

    if coords_bed_fn is None:  # All chromosomes
        chroms = utils.chroms_in_order

        start_pos_ = None
        end_pos_ = None

        coords_tuples = []
        for chrom in chroms:
            start_pos = 10_000 if start_pos_ is None else start_pos_
            end_pos = (
                utils.chrom_sizes_hg19_d[chrom] - 10_000
                if end_pos_ is None
                else end_pos_
            )

            coords_tuples.append((chrom, start_pos, end_pos))
    else:
        cdf = pd.read_csv(coords_bed_fn, sep="\t", header=None)
        coords_tuples = list(cdf.itertuples(index=False, name=None))

    if tmp_dir_path is None:
        tmp_dir_path = "tmp/"

    for cell_type in cell_types:
        logger.info(f"Currently processing {cell_type=}")
        GatingModel.generate_tracks_from_model(
            gm,
            cell_type=cell_type,
            coords_tuples=coords_tuples,
            combined_bigwigs_out_dir=combined_bigwigs_out_dir,
            tmp_dir_path=tmp_dir_path,
            part_length=part_length,
        )


def train_chromscorehmm(
    track_dir="tracks",
    num_states=15,
    out_dir="models/chromscorehmm",
    quantile=0.98,
    binarized_dir=None,
):
    from chromactivity.chromscorehmm import (
        chromhmm_learn_model,
        generate_binarized_directory,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_binarized_dir = tmpdir if binarized_dir is None else binarized_dir

        generate_binarized_directory(track_dir, out_binarized_dir, quantile=quantile)
        chromhmm_learn_model(
            bin_data_dir=out_binarized_dir,
            num_states=num_states,
            out_dir=out_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        prog="ChromActivity", description="Command line tool for ChromActivity"
    )

    subparsers = parser.add_subparsers(help="Sub-command help", dest="command")

    parser_train_experts = subparsers.add_parser("train_experts", help="Train experts")
    parser_train_experts.add_argument(
        "--labels_dir", default="data/labels", help="Labels directory"
    )
    parser_train_experts.add_argument(
        "--model_out_fn",
        default="models/chromactivity.model",
        help="Output model filename",
    )

    parser_generate_tracks = subparsers.add_parser(
        "generate_tracks", help="Generate tracks"
    )
    parser_generate_tracks.add_argument(
        "--model_fn", default="models/chromactivity.model", help="Model filename"
    )
    parser_generate_tracks.add_argument(
        "--cell_types", nargs="*", help="Cell types to process"
    )
    parser_generate_tracks.add_argument(
        "--combined_bigwigs_out_dir",
        default="tracks",
        help="Output directory for combined bigwigs",
    )
    parser_generate_tracks.add_argument(
        "--coords_bed_fn", default=None, help="Coordinate tuples bed"
    )
    parser_generate_tracks.add_argument(
        "--tmp_dir_path", default=None, help="Temporary directory path"
    )
    parser_generate_tracks.add_argument(
        "--part_length", type=int, default=1_000_000, help="Part length"
    )

    parser_train_chromscorehmm = subparsers.add_parser(
        "train_chromscorehmm", help="Train ChromScoreHMM"
    )
    parser_train_chromscorehmm.add_argument(
        "--track_dir", default="tracks", help="Track directory"
    )
    parser_train_chromscorehmm.add_argument(
        "--num_states", type=int, default=15, help="Number of states"
    )
    parser_train_chromscorehmm.add_argument(
        "--out_dir", default="models/chromscorehmm", help="Output directory"
    )
    parser_train_chromscorehmm.add_argument(
        "--quantile", type=float, default=0.98, help="Quantile"
    )
    parser_train_chromscorehmm.add_argument(
        "--binarized_dir", default=None, help="Binarized directory"
    )

    args = parser.parse_args()

    if args.command == "train_experts":
        train_experts(args.labels_dir, args.model_out_fn)
    elif args.command == "generate_tracks":
        generate_tracks(
            args.model_fn,
            args.cell_types,
            args.combined_bigwigs_out_dir,
            args.coords_bed_fn,
            args.tmp_dir_path,
            args.part_length,
        )
    elif args.command == "train_chromscorehmm":
        train_chromscorehmm(
            args.track_dir,
            args.num_states,
            args.out_dir,
            args.quantile,
            args.binarized_dir,
        )
    else:
        print("Command not recognized", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
