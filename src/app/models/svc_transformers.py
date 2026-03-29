import __main__

import numpy as np
import pandas as pd


def tenure_to_segment(col):
    """Bucket tenure values into the same categories used during training."""
    if isinstance(col, pd.DataFrame):
        series = col.iloc[:, 0]
    else:
        series = pd.Series(col.ravel())

    series = pd.to_numeric(series, errors="coerce").fillna(0)

    bins = [-np.inf, 12, 24, 48, np.inf]
    labels = ["New Customer", "Regular Customer", "Loyal Customer", "Very Loyal"]
    segmented = pd.cut(series, bins=bins, labels=labels, include_lowest=True)

    return pd.DataFrame({"tenure": segmented.astype(str)})


def map_yes_no_block(df_part):
    """Map Yes/No style categorical columns into numeric 1/0 values."""
    output = df_part.copy()
    for col in output.columns:
        output[col] = (
            output[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"no": 0.0, "yes": 1.0})
        )
    return output


def register_legacy_pickle_functions() -> None:
    """
    Register function names under __main__ so legacy notebook-pickled
    FunctionTransformer callables can be resolved by joblib.load.
    """
    setattr(__main__, "tenure_to_segment", tenure_to_segment)
    setattr(__main__, "map_yes_no_block", map_yes_no_block)
