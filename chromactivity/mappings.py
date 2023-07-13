import os

import pandas as pd

paths = {
    "wigToBigWig": "wigToBigWig",
    "bigWigToWig": "bigWigToWig",
    "chromsizes_hg19": "data/external/hg19.chrom.sizes",
}


class FileMapping:
    roadmap_bw_fn_template = (
        "data/raw/roadmap/signal/{eid}/{eid}-{mark}.imputed.pval.signal.bigwig"
    )

    # partial copy on scratch directory for faster access to bigwigs
    roadmap_bw_scratch_fn_template = (
        "scratch/data/raw/roadmap/signal/{eid}/{eid}-{mark}.imputed.pval.signal.bigwig"
    )

    roadmap_peak_fn_template = (
        "data/raw/roadmap/peaks/{eid}/{eid}-{mark}.imputed.narrowPeak.bed.nPk.gz"
    )

    roadmap_peak_fn_template_sorted_unzipped = (
        "data/raw/roadmap/peaks/{eid}/{eid}-{mark}.imputed.narrowPeak.bed.nPk"
    )

    @classmethod
    def get_signal_fn(cls, mark, cell_type, roadmap_or_epimap="roadmap"):
        if roadmap_or_epimap == "roadmap":
            return cls.get_roadmap_signal_fn(mark, cell_type)

    @classmethod
    def get_peak_fn(cls, mark, cell_type, roadmap_or_epimap="roadmap", is_sorted=True):
        if roadmap_or_epimap == "roadmap":
            return cls.get_roadmap_peak_fn(mark, cell_type, is_sorted=is_sorted)

    @classmethod
    def get_roadmap_signal_fn(cls, mark: str, cell_type: str):
        eid = IdMapping.to_eid(cell_type)

        assert (
            eid is not None and eid[0] == "E"
        ), f"Invalid cell type: {cell_type} -> {eid}"
        assert mark in MarksMapping.roadmap_marks, f"Invalid mark: {mark}"

        fn = cls.roadmap_bw_scratch_fn_template.format(eid=eid, mark=mark)
        if os.path.exists(fn):
            return fn
        else:
            return cls.roadmap_bw_fn_template.format(eid=eid, mark=mark)

    @classmethod
    def get_roadmap_peak_fn(cls, mark: str, cell_type: str, is_sorted=True):
        eid = IdMapping.to_eid(cell_type)

        assert (
            eid is not None and eid[0] == "E"
        ), f"Invalid cell type: {cell_type} -> {eid}"
        assert mark in MarksMapping.roadmap_marks, f"Invalid mark: {mark}"

        if is_sorted:
            return cls.roadmap_peak_fn_template_sorted_unzipped.format(
                eid=eid, mark=mark
            )
        else:
            return cls.roadmap_peak_fn_template.format(eid=eid, mark=mark)


class IdMapping:
    @classmethod
    def to_eid(cls, cell_type):
        assert cell_type[0] == "E"
        return cell_type


class ChromStateMapping:
    chrom_state_mapping_d = {
        "chrom_state_15": {
            1: "1_TssA",
            2: "2_TssAFlnk",
            3: "3_TxFlnk",
            4: "4_Tx",
            5: "5_TxWk",
            6: "6_EnhG",
            7: "7_Enh",
            8: "8_ZNF/Rpts",
            9: "9_Het",
            10: "10_TssBiv",
            11: "11_BivFlnk",
            12: "12_EnhBiv",
            13: "13_ReprPC",
            14: "14_ReprPCWk",
            15: "15_Quies",
        },
        "chrom_state_18": {
            1: "1_TssA",
            2: "2_TssFlnk",
            3: "3_TssFlnkU",
            4: "4_TssFlnkD",
            5: "5_Tx",
            6: "6_TxWk",
            7: "7_EnhG1",
            8: "8_EnhG2",
            9: "9_EnhA1",
            10: "10_EnhA2",
            11: "11_EnhWk",
            12: "12_ZNF/Rpts",
            13: "13_Het",
            14: "14_TssBiv",
            15: "15_EnhBiv",
            16: "16_ReprPC",
            17: "17_ReprPCWk",
            18: "18_Quies",
        },
        "chrom_state_25": {
            1: "1_TssA",
            2: "2_PromU",
            3: "3_PromD1",
            4: "4_PromD2",
            5: "5_Tx5'",
            6: "6_Tx",
            7: "7_Tx3'",
            8: "8_TxWk",
            9: "9_TxReg",
            10: "10_TxEnh5'",
            11: "11_TxEnh3'",
            12: "12_TxEnhW",
            13: "13_EnhA1",
            14: "14_EnhA2",
            15: "15_EnhAF",
            16: "16_EnhW1",
            17: "17_EnhW2",
            18: "18_EnhAc",
            19: "19_DNase",
            20: "20_ZNF/Rpts",
            21: "21_Het",
            22: "22_PromP",
            23: "23_PromBiv",
            24: "24_ReprPC",
            25: "25_Quies",
        },
    }

    @classmethod
    def get_chromstate_order(cls, cs_annot):
        return list(cls.chrom_state_mapping_d[cs_annot].values())

    @classmethod
    def get_chromstate_annot_fn(
        cls, chromstate: str, cell_type: str, roadmap_or_epimap: str
    ):
        chrom_state_annot_fn_d_scratch = {
            "chrom_state_25": "data/raw/roadmap/chromstate/chromstate_25/{EID}/{EID}_25_imputed12marks_mnemonics.sorted.bed",
            "chrom_state_15": "data/raw/roadmap/chromstate/chromstate_15/{EID}/{EID}_15_coreMarks_mnemonics.sorted.bed",
            "chrom_state_18": "data/raw/roadmap/chromstate/chromstate_18/{EID}/{EID}_18_core_K27ac_mnemonics.sorted.bed",
        }

        chrom_state_annot_fn_d = {
            "chrom_state_25": "data/raw/roadmap/chromstate/chromstate_25/{EID}/{EID}_25_imputed12marks_mnemonics.sorted.bed",
            "chrom_state_15": "data/raw/roadmap/chromstate/chromstate_15/{EID}/{EID}_15_coreMarks_mnemonics.sorted.bed",
            "chrom_state_18": "data/raw/roadmap/chromstate/chromstate_18/{EID}/{EID}_18_core_K27ac_mnemonics.sorted.bed",
        }

        if roadmap_or_epimap == "roadmap":
            eid = IdMapping.to_eid(cell_type)
            fn = chrom_state_annot_fn_d_scratch[chromstate].format(EID=eid)
            if os.path.exists(fn):
                return fn
            else:
                return chrom_state_annot_fn_d[chromstate].format(EID=eid)
        else:
            raise ValueError(f"Invalid: {roadmap_or_epimap}")

    chrom_state_colors_d = {
        "chrom_state_15": {
            "1_TssA": "#ff0000",
            "2_TssAFlnk": "#ff4500",
            "3_TxFlnk": "#32cd32",
            "4_Tx": "#008000",
            "5_TxWk": "#006400",
            "6_EnhG": "#c2e105",
            "7_Enh": "#ffff00",
            "8_ZNF/Rpts": "#66cdaa",
            "9_Het": "#8a91d0",
            "10_TssBiv": "#cd5c5c",
            "11_BivFlnk": "#e9967a",
            "12_EnhBiv": "#bdb76b",
            "13_ReprPC": "#808080",
            "14_ReprPCWk": "#c0c0c0",
            "15_Quies": "#ffffff",
        },
        "chrom_state_18": {
            "1_TssA": "#ff0000",
            "2_TssFlnk": "#ff4500",
            "3_TssFlnkU": "#ff4500",
            "4_TssFlnkD": "#ff4500",
            "5_Tx": "#008000",
            "6_TxWk": "#006400",
            "7_EnhG1": "#c2e105",
            "8_EnhG2": "#c2e105",
            "9_EnhA1": "#ffc34d",
            "10_EnhA2": "#ffc34d",
            "11_EnhWk": "#ffff00",
            "12_ZNF/Rpts": "#66cdaa",
            "13_Het": "#8a91d0",
            "14_TssBiv": "#cd5c5c",
            "15_EnhBiv": "#bdb76b",
            "16_ReprPC": "#808080",
            "17_ReprPCWk": "#c0c0c0",
            "18_Quies": "#ffffff",
        },
        "chrom_state_25": {
            "1_TssA": "#ff0000",
            "2_PromU": "#ff4500",
            "3_PromD1": "#ff4500",
            "4_PromD2": "#ff4500",
            "5_Tx5'": "#008000",
            "6_Tx": "#008000",
            "7_Tx3'": "#008000",
            "8_TxWk": "#009600",
            "9_TxReg": "#c2e105",
            "10_TxEnh5'": "#c2e105",
            "11_TxEnh3'": "#c2e105",
            "12_TxEnhW": "#c2e105",
            "13_EnhA1": "#ffc34d",
            "14_EnhA2": "#ffc34d",
            "15_EnhAF": "#ffc34d",
            "16_EnhW1": "#ffff00",
            "17_EnhW2": "#ffff00",
            "18_EnhAc": "#ffff00",
            "19_DNase": "#ffff66",
            "20_ZNF/Rpts": "#66cdaa",
            "21_Het": "#8a91d0",
            "22_PromP": "#e6b8b7",
            "23_PromBiv": "#7030a0",
            "24_ReprPC": "#808080",
            "25_Quies": "#ffffff",
        },
    }

    @classmethod
    def get_chromstate_genome_coverage_for_enrichment(
        cls, chromstate_annot, cell_type, roadmap_or_epimap
    ):
        chromstate_annot_fn = cls.get_chromstate_annot_fn(
            chromstate=chromstate_annot,
            cell_type=cell_type,
            roadmap_or_epimap=roadmap_or_epimap,
        )

        dz = pd.read_csv(
            chromstate_annot_fn,
            sep="\t",
            names=["chrom", "start", "end", "chromstate"],
            dtype={"chrom": str, "start": "int32", "end": "int32", "chromstate": str},
        )

        dz["len"] = dz["end"] - dz["start"]
        genome_background = dz.groupby("chromstate")["len"].sum() / dz["len"].sum()

        return genome_background.rename(chromstate_annot)

    chromscore_hmm_state_map = {
        1: "E10",
        2: "E13",
        3: "E11",
        4: "E6",
        5: "E2",
        6: "E7",
        7: "E15",
        8: "E9",
        9: "E14",
        10: "E12",
        11: "E3",
        12: "E8",
        13: "E1",
        14: "E5",
        15: "E4",
    }

    chromscore_hmm_state_map_inv = {v: k for k, v in chromscore_hmm_state_map.items()}

    chromscore_hmm_color_map = {
        1: "#ff0000",
        2: "orangered",
        3: "orangered",
        4: "orangered",
        5: "darkorange",
        6: "#8a91d0",
        7: "#f9d938",
        8: "#f9d938",
        9: "#f9d938",
        10: "khaki",
        11: "#e6b8b7",
        12: "#e6b8b7",
        13: "#e6b8b7",
        14: "#e6b8b7",
        15: "#ffffff",
    }


class MarksMapping:
    roadmap_marks = [
        "DNase",
        "H2A.Z",
        "H3K27ac",
        "H3K27me3",
        "H3K36me3",
        "H3K4me1",
        "H3K4me2",
        "H3K4me3",
        "H3K79me2",
        "H3K9ac",
        "H3K9me3",
        "H4K20me1",
    ]


class PlotMapping:
    label_mapping_datasets = {
        "crispr-k562-fulco": "Fulco (CRISPR/K562)",
        "crispr-k562-gasperini_pval": "Gasperini (CRISPR/K562)",
        "starr-gm-wang": "Wang (STARR/GM12878)",
        "mpra-hepg2-ernst": "Ernst (MPRA/HepG2)",
        "mpra-hepg2-kheradpour_pval": "Kheradpour (MPRA/HepG2)",
        "mpra-k562-ernst": "Ernst (MPRA/K562)",
        "mpra-k562-kheradpour_pval": "Kheradpour (MPRA/K562)",
        "starr-a549-white": "White (STARR/A549)",
        "starr-helaS3-muerdter": "Muerdter (STARR/HeLaS3)",
        "starr-hepg2-white": "White (STARR/HepG2)",
        "starr-k562-white": "White (STARR/K562)",
    }
