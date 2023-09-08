import pandas as pd
import numpy as np

def pandas_to_sep005(
        dataframe: pd.DataFrame,
        name: str,
        unit_str: str,
        additional_info: dict = {}
    ):
    """
    Convert pandas dataframe to sep005 format
    :param dataframe: pandas dataframe
    :return: sep005 format
    """
    sep005 = {}
    sep005['data'] = \
        np.vstack(
            [dataframe[column].values for column in dataframe.columns] # type: ignore
        )
    sep005['fs'] = \
        np.round(
            len(dataframe.index) \
            / (
                dataframe.index[-1] - dataframe.index[0]
            ).total_seconds()
        )

    sep005['name'] = name
    sep005['time'] = \
        np.vstack(
            [dataframe.index.values for column in dataframe.columns]
        )
    sep005["channel_names"] = dataframe.columns
    sep005["unit_str"] = unit_str
    for info in additional_info:
        sep005[info] = additional_info[info]
    return sep005