import pandas as pd

from chromactivity import mappings

feature_transform_id = "ft_20220505_cs25"


class SignalScaling:
    @classmethod
    def get_transformed_signal_df(cls, raw_feat_df, cell_type=None):
        cols = raw_feat_df.columns.get_level_values(0).str.startswith("feat_signal_")
        return raw_feat_df.loc(axis=1)[cols]


class SignalProcessing:
    @classmethod
    def get_maxsignal_df(cls, raw_feat_df):
        feat_max_d = {}
        for mark in mappings.MarksMapping.roadmap_marks:
            feat_ = f"feat_signal_{mark}"
            feat_df = raw_feat_df.loc[:, feat_]
            feat_max_d[(f"feat_maxsignal_{mark}", 0)] = feat_df.max(axis=1)

        maxsignal_df = pd.concat(feat_max_d, axis=1)
        return maxsignal_df

    @classmethod
    def get_signal_at_offset_zero(cls, raw_feat_df):
        feat_d = {}
        for mark in mappings.MarksMapping.roadmap_marks:
            feat_ = f"feat_signal_{mark}"
            feat_at_zero = raw_feat_df.loc(axis=1)[feat_, 0]
            feat_d[(f"feat_offsetzero_{mark}", 0)] = feat_at_zero

        fdf = pd.concat(feat_d, axis=1)
        return fdf


class PeakProcessing:
    @classmethod
    def get_peak_df(cls, raw_feat_df):
        mask = raw_feat_df.columns.get_level_values(0).str.startswith("feat_peak_")
        return raw_feat_df.loc[:, mask]


class ChromatinStateProcessing:
    @classmethod
    def get_one_hot_encoded_chromstate(cls, chrom_state_series, chrom_state_annot=None):
        if chrom_state_annot is None:
            if "15_Quies" in chrom_state_series.values:
                chrom_state_annot = "chrom_state_15"
            elif "18_Quies" in chrom_state_series.values:
                chrom_state_annot = "chrom_state_18"
            elif "25_Quies" in chrom_state_series.values:
                chrom_state_annot = "chrom_state_25"
            elif "Quies" in chrom_state_series.values:
                chrom_state_annot = "chrom_state_18_epimap"
            else:
                raise ValueError("Could not figure out chrom_state_annot.")

        cs_order = mappings.ChromStateMapping.get_chromstate_order(chrom_state_annot)

        from sklearn.preprocessing import OneHotEncoder

        ohe = OneHotEncoder(categories=[cs_order], sparse=False)
        colnames = chrom_state_annot + "." + pd.Series(cs_order)

        cs_ = pd.DataFrame(chrom_state_series)

        cs_one_hot_df = pd.DataFrame(
            ohe.fit_transform(cs_),
            index=chrom_state_series.index,
            columns=colnames,
        ).applymap(int)

        return cs_one_hot_df

    @classmethod
    def get_one_hot_encoded_chromstate_df(cls, raw_feat_df, chrom_state_annot):
        feat_chrom_state_name = f"feat_{chrom_state_annot}"
        chrom_state_series = raw_feat_df.loc[:, (feat_chrom_state_name, 0)]
        chrom_state_series = chrom_state_series.rename(chrom_state_annot)
        cs_df = cls.get_one_hot_encoded_chromstate(
            chrom_state_series, chrom_state_annot
        )
        # add 0s to column index
        cs_df.columns = pd.MultiIndex.from_product([cs_df.columns, [0]])
        return cs_df


class TssDistProcessing:
    pass


def get_transformed_raw_feat_df(
    raw_feat_df, cell_type, chrom_state_annot="chrom_state_25"
):
    transformed_signal_df = SignalScaling.get_transformed_signal_df(
        raw_feat_df, cell_type
    )
    # maxsignal_df = SignalProcessing.get_maxsignal_df(transformed_signal_df)
    signal_offsetzero_df = SignalProcessing.get_signal_at_offset_zero(
        transformed_signal_df
    )
    peak_df = PeakProcessing.get_peak_df(raw_feat_df)

    if chrom_state_annot is None:
        cs_df = None
    else:
        cs_df = ChromatinStateProcessing.get_one_hot_encoded_chromstate_df(
            raw_feat_df, chrom_state_annot
        )

    feat_df = pd.concat(
        [
            transformed_signal_df,
            # maxsignal_df,
            signal_offsetzero_df,
            peak_df,
            cs_df,
        ],
        axis=1,
    )

    return feat_df


def get_transformed_X(X, cell_type, chrom_state_annot="chrom_state_25"):
    from chromactivity.dataset import Dataset

    rdf = Dataset._X_to_raw_feat_df(X)
    rdf_ = get_transformed_raw_feat_df(
        rdf, cell_type=cell_type, chrom_state_annot=chrom_state_annot
    )
    return Dataset._raw_feat_df_to_X(rdf_)


def get_transformed_dataset(ds, copy=True, chrom_state_annot="chrom_state_25"):
    transformed_raw_feat_df = get_transformed_raw_feat_df(
        ds.raw_feat_df,
        mappings.IdMapping.to_eid(ds.cell_type),
        chrom_state_annot=chrom_state_annot,
    )
    ds_ = ds.copy() if copy else ds
    ds_._raw_feat_df = transformed_raw_feat_df
    ds_.feature_transform_id = feature_transform_id
    ds_.feat_table_fn = None
    return ds_


def get_transformer(feature_transform_id, kind="X"):
    if feature_transform_id == "ft_20220505_cs25":
        if kind == "raw_feat_df":
            return get_transformed_raw_feat_df
        elif kind == "X":
            return get_transformed_X
        elif kind == "dataset":
            return get_transformed_dataset
        else:
            raise ValueError("Invalid kind")
    else:
        raise ValueError(f"Unknown feature transform: {feature_transform_id}")
