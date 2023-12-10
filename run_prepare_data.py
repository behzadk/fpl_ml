import os
import re
from glob import glob

import pandas as pd
from loguru import logger

from ml_core.utils import add_prefix_to_columns, get_columns_with_prefix
from user_config import PROCESSED_DATA_DIR, VAASTAV_FPL_DIR


def _get_gw_paths(season_dir: os.PathLike) -> pd.DataFrame:
    """Gets paths to each game week csv from the season directory. Generates a
    dataframe with columns designating the path to the `.csv`

    +---------------------------+-----------+---------+
    |      game_week_path       | game_week | season  |
    +---------------------------+-----------+---------+
    | /data/2018-19/gws/gw1.csv |         1 | 2018-19 |
    | ...                       |       ... | ...     |
    +---------------------------+-----------+---------+


    Args:
        season_dir: directory to a season. e.g /data/2018-19/

    Returns:
        Dataframe containing paths to all the game weeks in the provided `season_dir`
    """

    gw_dir = f"{season_dir}/gws/"
    gw_file_pattern = r"gw(\d+)\.csv"

    data = {"game_week": [], "game_week_path": []}

    for file_path in glob(f"{gw_dir}/*.csv"):
        file_name = file_path.split("/")[-1]
        match = re.search(gw_file_pattern, file_name)
        if match:
            gw_number = match.group(1)
            data["game_week"].append(gw_number)
            data["game_week_path"].append(file_path)

    df = pd.DataFrame.from_dict(data)
    df["season"] = season_dir.split("/")[-2]

    return df


def _merge_additional_raw_data(
    gw_df: pd.DataFrame,
    player_raw_df: pd.DataFrame,
    player_id_df,
    master_team_df: pd.DataFrame,
    season: str,
):
    """Merges player positions found under `element_type`, team id `team` and
    team name (`team_name`) from the raw player data of the season.

    Args:
        gw_df: Dataframe for a game week, where each row represents a player
        season: The season from which the game week comes from.
        master_team_df: master team data frame containing team ids and team names
        data_dir: Directory of the
    """

    season_team_df = master_team_df.loc[master_team_df["season"] == season]

    if "player_id" not in player_raw_df.columns:
        player_raw_df = player_raw_df.rename_axis("player_id").reset_index()
        player_raw_df["player_id"] = player_raw_df["player_id"].astype(str)

    player_raw_df = player_raw_df[["player_id", "element_type", "team"]]

    raw_data_df = pd.merge(player_raw_df, season_team_df, on="team")
    gw_df = pd.merge(gw_df, raw_data_df, on="player_id")

    return gw_df


def _generate_gw_paths_df(data_dir):
    # Use list comprehension to generate a list of DataFrames
    game_week_paths_df = pd.concat(
        [
            pd.DataFrame(_get_gw_paths(season_dir))
            for season_dir in glob(f"{data_dir}/**/")
        ]
    )

    game_week_paths_df["game_week"] = game_week_paths_df["game_week"].astype(int)
    game_week_paths_df = game_week_paths_df.sort_values(
        by=["season", "game_week"], ascending=True
    )

    return game_week_paths_df


def _generate_player_summary_statistics(master_df: pd.DataFrame):
    """Each row of the dataframe contains stats for how the player performed on
    a given game week, in a given season.

    This function generates summary statistics that describe how the player has performed up to the given game week.

    Args:
        master_df: Dataframe where each row is a player's performance for a single game week
    """

    player_dfs = []
    for player_name in master_df.player_name.unique():
        logger.info(f"Preparing summary statistics for {player_name}")
        player_df = master_df.loc[master_df.player_name == player_name]

        player_df = player_df.sort_values(by=["starting_year", "game_week"])

        window_sizes = [2, 4, 6, 8]
        stat_columns = ["minutes", "assists", "goals_conceded", "goals_scored", "bonus"]
        for w in window_sizes:
            for s in stat_columns:
                player_df = generate_moving_average(
                    player_df, column=s, window_size=w, fill_value=-1
                )

        player_dfs.append(player_df)

    return pd.concat(player_dfs)


def _load_gw_data(game_week_path, game_week):
    gw_df = pd.read_csv(game_week_path, encoding="latin-1")

    gw_df["game_week"] = game_week

    gw_df = gw_df.rename(columns={"name": "player_name"})

    if len(gw_df) == 0:
        return None

    if "element" not in gw_df.columns:
        gw_df["player_id"] = gw_df["name"].apply(lambda n: str(n.split("_")[-1]))

    else:
        gw_df = gw_df.rename(columns={"element": "player_id"})
        gw_df["player_id"] = gw_df["player_id"].astype(str)

    return gw_df


def _clean_player_names(df):
    # Remove the player id suffix if it exists.
    df["player_name"] = df.apply(
        lambda row: row["player_name"].replace(f"_{str(row.player_id)}", ""), axis=1
    )

    # Replace white space with undescores
    df["player_name"] = df.apply(
        lambda row: row["player_name"].replace(" ", "_"), axis=1
    )

    # Make everything lower case
    df["player_name"] = df.apply(lambda row: row["player_name"].lower(), axis=1)

    return df


def _clean_game_weeks(df, game_week_column, season_column):
    clean_dfs = []
    for season in df[season_column].unique():
        season_df = df.loc[df[season_column] == season]
        season_game_weeks = season_df[game_week_column].unique()
        season_game_weeks = sorted(season_game_weeks)

        remapped_game_weeks = {}

        for i in range(len(season_game_weeks) - 1):
            if season_game_weeks[i + 1] - season_game_weeks[i] > 1:
                remapped_game_weeks[season_game_weeks[i + 1]] = season_game_weeks[i] + 1
                season_game_weeks[i + 1] = season_game_weeks[i] + 1

        season_df[game_week_column] = season_df[game_week_column].replace(
            remapped_game_weeks
        )

        clean_dfs.append(season_df)

    return pd.concat(clean_dfs)


def generate_moving_average(player_season_df, column, window_size, fill_value=None):
    player_season_df[f"rolling_mean_{window_size}_{column}"] = (
        player_season_df[column]
        .rolling(window=window_size)
        .mean()
        .shift(fill_value=fill_value)
    )
    player_season_df[f"rolling_mean_{window_size}_{column}"] = player_season_df[
        f"rolling_mean_{window_size}_{column}"
    ].fillna(fill_value)

    return player_season_df


def get_season_start_year(year_string):
    start_year = int(year_string.split("-")[0])

    # Return the start year for sorting
    return start_year


def main():
    output_data_dir = PROCESSED_DATA_DIR
    raw_data_dir = os.path.join(VAASTAV_FPL_DIR, 'data')

    os.makedirs(output_data_dir, exist_ok=True)

    master_team_df = pd.read_csv(os.path.join(raw_data_dir, "master_team_list.csv"))
        
    master_team_df = master_team_df.rename(columns={"id": "team_id"})

    # Use list comprehension to generate a list of DataFrames
    game_week_paths_df = _generate_gw_paths_df(raw_data_dir)

    gw_dataframes = []

    for idx, row in game_week_paths_df.iterrows():
        season = row["season"]

        logger.info(f"Loading gw: {row['game_week']}, season: {season}")
        gw_df = _load_gw_data(
            game_week_path=row["game_week_path"], game_week=row["game_week"]
        )

        if gw_df is None:
            continue

        player_raw_df = pd.read_csv(f"{raw_data_dir}/{season}/players_raw.csv")
        player_id_df = pd.read_csv(f"{raw_data_dir}/{season}/player_idlist.csv")
        player_id_df = player_id_df.rename(columns={"id": "player_id"})

        logger.info(f"Preparing gw: {row['game_week']}, season: {season}")

        gw_df = _merge_additional_raw_data(
            gw_df,
            player_raw_df=player_raw_df,
            player_id_df=player_id_df,
            master_team_df=master_team_df,
            season=row["season"],
        )

        gw_dataframes.append(gw_df)

    master_df = pd.concat(gw_dataframes)
    master_df = _clean_player_names(master_df)
    master_df = _clean_game_weeks(
        master_df, game_week_column="game_week", season_column="season"
    )

    master_df["starting_year"] = master_df.apply(
        lambda row: get_season_start_year(row["season"]), axis=1
    )

    X = [
        "value",
        "element_type",
        "opponent_team",
        "game_week",
        "team_name",
        "was_home",
    ]
    y = ["total_points"]

    rolling_mean_columns = get_columns_with_prefix(master_df, prefix_list=['rolling_mean_'])
    X.extend(rolling_mean_columns)

    master_df = _generate_player_summary_statistics(master_df)

    # Holdout year for test set
    test_year = 2022
    test_df = master_df.loc[master_df['starting_year'] == test_year][[*X, *y]]

    train_df = master_df.loc[master_df['starting_year'] != test_year][[*X, *y]]

    test_df = add_prefix_to_columns(test_df, columns=X, prefix="X_")
    train_df = add_prefix_to_columns(train_df, columns=X, prefix="X_")


    train_df.to_csv(os.path.join(output_data_dir, 'train_val_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_data_dir, 'test_data.csv'), index=False)

    master_df.to_csv(os.path.join(output_data_dir, 'master.csv'), index=False)

if __name__ == "__main__":
    main()
