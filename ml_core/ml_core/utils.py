import time
from typing import Optional, cast

import pandas as pd
from lightning.pytorch import seed_everything
import mlflow
import shutil


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



def get_experiment_runs(
    experiment_name: str,
    tracking_uri: str,
) -> pd.DataFrame:
    """Gets all runs of an experiment.
    Args:
        experiment_name (str): Name of MLflow experiment.
        tracking_uri (str): The URI for the MLflow tracking server.
    Returns:
        Dataframe of MLflow runs.
    """

    mlflow.set_tracking_uri(tracking_uri)

    current_experiment = mlflow.get_experiment_by_name(experiment_name)

    runs_df = mlflow.search_runs([current_experiment.experiment_id])

    return runs_df

def delete_runs_by_metric(mlruns_dir, experiment_name, keep_n_runs=25, metric='val_MSE', ascending=True):
    """Permanently deletes runs after sorting by a given metric. 

    e.g `keep_n_runs=25, metric='val_MSE', ascending=True` will keep the runs with the lowest `val_MSE` scores

    Args:
        mlruns_dir: MLflow runs dir.
        experiment_name: Experiment name.
        keep_n_runs: Number of runs to keep after sorting. Defaults to 25.
    """
    
    def remove_run_dir(run_dir):
        shutil.rmtree(run_dir, ignore_errors=True)

    runs_df = get_experiment_runs(tracking_uri=f"file://{mlruns_dir}", experiment_name=experiment_name)
    runs_df = runs_df.sort_values(by=f'metrics.{metric}', ascending=True)
    
    if keep_n_runs < runs_df.shape[0]:
        drop_runs = runs_df.tail(runs_df.shape[0] - keep_n_runs)

        for r in drop_runs.run_id:
            mlflow.delete_run(run_id=r)
            remove_run_dir(f"{mlruns_dir}/{r}/")
