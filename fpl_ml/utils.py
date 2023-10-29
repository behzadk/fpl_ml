import time
from typing import Optional, cast

import pandas as pd
from lightning.pytorch import seed_everything


def set_random_seeds(random_seed: Optional[int]) -> int:
    """Sets random seed using provided integer.

    If argument is None, generates a pseudo random seed
    to use.

    Args:
        random_seed (Union[int, None]): Seed to use, or None to use pseudo random seed

    Returns:
        int: Seed used to set everything
    """

    if not random_seed:
        t = 1000 * time.time()  # current time in milliseconds
        random_seed = int(t) % 2**32

    random_seed = cast(int, random_seed)

    seed_everything(random_seed)

    return random_seed


def get_columns_with_prefix(df: pd.DataFrame, prefix_list: list[str]) -> list[str]:
    """Returns columns of the dataframe that start with a provided prefix

    Args:
        df (pd.DataFrame): dataframe
        prefix_list (list[str]): prefixes used to extract columns

    Returns:
        list[str]: list of columns starting with prefix
    """
    return list(df.columns[df.columns.str.startswith(tuple(prefix_list))])


def add_prefix_to_columns(df, columns, prefix):
    new_names = {i: prefix + i for i in columns}
    df = df.rename(columns=new_names)

    return df
