from math import e
from fpl_ml_projects.fpl_ml.constants import (
    Positions,
    CategoricalFeatures,
    NumericalFeatures,
    Labels
)
from fpl_ml_projects.fpl_ml.simulation.user import UserTeam
from fpl_ml_projects.fpl_ml.simulation.player import Player
import pandas as pd
import numpy as np
from loguru import logger
from torch import Value


def make_player_pool(gw_df, points_column=None):
    """
    Make a player pool from the gw_df
    """
    player_pool = []
    for _, row in gw_df.iterrows():
        player = Player(
                name=row["player_name"],
                position=row[CategoricalFeatures.ELEMENT_TYPE],
                value=row[NumericalFeatures.VALUE],
                club=row[CategoricalFeatures.TEAM_NAME],
            )

        if points_column:
            player.points = row[points_column]

        player_pool.append(player)

    return player_pool


def sample_legal_team(
    season_df,
    name_column="player_name",
    position_column=CategoricalFeatures.ELEMENT_TYPE,
    value_column=NumericalFeatures.VALUE,
    club_column=CategoricalFeatures.TEAM_NAME,
    points_column=None,
):
    """
    Sample a legal team from the season_df
    """

    keep_columns = [name_column, position_column, value_column, club_column, points_column]
    keep_columns = [c for c in keep_columns if c is not None]
    season_df = season_df[
       keep_columns
    ].drop_duplicates()
    sampled_players = []
    sampled_player_names = []

    player_pool = make_player_pool(season_df, points_column=points_column)

    def _remove_illegal_clubs(player_pool, all_sampled_player):
        tmp_team = UserTeam(all_sampled_player)
        player_club_count = tmp_team.count_players_from_each_club()
        illegal_clubs = [club for club, count in player_club_count.items() if count > 2]

        return [p for p in player_pool if p.club not in illegal_clubs]
    
    def _sample_player(player_pool, position, all_player_names, all_sampled_player):
        sub_player_pool = [p for p in player_pool if p.position == position]
        sub_player_pool = _remove_illegal_clubs(sub_player_pool, all_sampled_player)

        player = np.random.choice(sub_player_pool)

        all_player_names.append(player.name)
        all_sampled_player.append(player)

        return player

    # Pick two goalkeepers
    for i in range(2):
        player = _sample_player(
            player_pool, Positions.GOALKEEPER, sampled_player_names, sampled_players
        )
        player_pool.remove(player)

    # Pick five defenders from different clubs
    for i in range(5):
        player = _sample_player(
            player_pool, Positions.DEFENDER, sampled_player_names, sampled_players
        )
        player_pool.remove(player)
    # Pick five midfielders from different clubs
    for i in range(5):
        player = _sample_player(
            player_pool, Positions.MIDFIELDER, sampled_player_names, sampled_players
        )
        player_pool.remove(player)

    # Pick three forwards from different clubs
    for i in range(3):
        player = _sample_player(
            player_pool, Positions.FORWARD, sampled_player_names, sampled_players
        )
        player_pool.remove(player)

    # Check if team value is legal
    team = UserTeam(sampled_players)
    while not team.check_team_legality():
        # Get most expensive player
        expensive_player = sorted(team.players, key=lambda x: x.value, reverse=True)[0]

        # Transfer the player for a cheaper player
        cheaper_player_pool = [p for p in player_pool if p.value < expensive_player.value]
        cheaper_player = _sample_player(
            cheaper_player_pool,
            expensive_player.position, 
            sampled_player_names, 
            sampled_players
        )

        # Transfer the player
        team.transfer_player(cheaper_player, expensive_player)

        # Remove the player from the season_df
        player_pool.remove(cheaper_player)

        # Add the expensive player back to the pool
        player_pool.append(expensive_player)

    return team, player_pool


class BaseAgent:
    def __init__(self, name: str, team: UserTeam):
        self.team = team
        self.name = name

    def init_team(self, gw_df: pd.DataFrame):
        """Initializes a legal team given a starting gameweek

        Args:
            gw_df: DataFrame containing a gameweek

        Raises:
            ValueError: If the team is not legal
        """
        self.team, _ = sample_legal_team(gw_df)

        if not self.team.check_team_legality():
            positions_legal = self.team._check_legality_of_player_positions()
            team_value_legal = self.team._check_legality_of_team_value()
            players_from_same_club_legal = (
                self.team._check_legality_of_team_players_from_same_club()
            )

            raise ValueError(
                f"Team is not legal, positions_legal={positions_legal}, team_value_legal={team_value_legal}, players_from_same_club_legal={players_from_same_club_legal}"
            )


class HillClimbingAgent(BaseAgent):
    def __init__(self, name: str, team: UserTeam, n_iterations: int = 100, n_restarts: int = 10, points_column: str = Labels.TOTAL_POINTS):
        super().__init__(name, team)

        self.n_iterations = n_iterations
        self.n_restarts = n_restarts
        self.points_column = points_column

    def init_team(self, gw_df: pd.DataFrame):
        """Initializes a legal team given a starting gameweek

        Args:
            gw_df: DataFrame containing a gameweek

        Raises:
            ValueError: If the team is not legal
        """
        # Initialize player pool

        def calculate_total_team_points(team: UserTeam) -> float:
            player_points = [p.points for p in team.players]
            return np.sum(player_points)

        best_team_points = 0
        best_team = None

        for _ in range(self.n_restarts):
            current_team, player_pool = sample_legal_team(gw_df, points_column=self.points_column)
            for x in range(self.n_iterations):
                    
                # Randomly select player from team
                transfer_out_player = np.random.choice(current_team.players)

                # Get players in removed player's position
                sub_player_pool = [p for p in player_pool if p.position == transfer_out_player.position]

                # Subset for players with a higher points
                sub_player_pool = [p for p in sub_player_pool if p.points > transfer_out_player.points]

                if len(sub_player_pool) == 0:
                    continue

                # Randomly select player from sub player pool
                transfer_in_player = np.random.choice(sub_player_pool)

                new_team = current_team.clone()
                new_team.transfer_player(player_in=transfer_in_player, player_out=transfer_out_player)

                if new_team.check_team_legality():
                    player_pool.append(transfer_out_player)
                    player_pool.remove(transfer_in_player)                    
                    current_team = new_team
                
                else:
                    continue


                if (total_points := calculate_total_team_points(current_team)) > best_team_points:
                    best_team_points = total_points
                    best_team = current_team


        self.team = best_team

    def step_team(self, gw_df: pd.DataFrame, n_free_transfers: int = 1):
        """Perform a step in the environment

        Args:
            gw_df: DataFrame containing a gameweek

        Returns:
            observation: new observation
            reward_n: list of rewards
            done: boolean indicating if the season is over
        """

        def calculate_total_team_points(team: UserTeam, n_free_transfers, n_transfers) -> float:
            player_points = [p.points for p in team.players]
            
            # Calculate number of penalty transfers
            n_penalty_transfers = max(n_transfers - n_free_transfers, 0)

            # Points penalty
            points_penalty = n_penalty_transfers * 2

            return np.sum(player_points) - points_penalty

        def count_number_of_transfers(team_a, team_b):
            return len([p for p in team_a.players if not any(p == item for item in team_b.players)])

        def update_player_points(players: list[Player], player_pool: list[Player]):

            # Update the player gw points to those of the player pool
            for player in players:
                if player not in player_pool:
                    player.points = 0.0

                else:
                    matching_pool_player = player_pool[player_pool.index(player)]

                    player.points = matching_pool_player.points


        for _ in range(self.n_restarts):
            # Create player pool
            player_pool = make_player_pool(gw_df, points_column=self.points_column)

            # Replace starting team players with players from the player pool
            update_player_points(self.team.players, player_pool)

            # Remove any starting team players from the pool
            player_pool = [p for p in player_pool if not any(p == item for item in self.team.players)]

            best_team_points = calculate_total_team_points(self.team, n_free_transfers=1, n_transfers=0)
            best_team = self.team
            

            current_team = self.team.clone()

            for x in range(self.n_iterations):

                # Randomly select player from current team
                transfer_out_player = np.random.choice(current_team.players)

                # Get players in removed player's position
                sub_player_pool = [p for p in player_pool if p.position == transfer_out_player.position]

                # Substet for players with a higher points
                sub_player_pool = [p for p in sub_player_pool if p.points > transfer_out_player.points]

                if len(sub_player_pool) == 0:
                    continue

                # Randomly select player from sub player pool
                transfer_in_player = np.random.choice(sub_player_pool)

                new_team = current_team.clone()
                new_team.transfer_player(player_in=transfer_in_player, player_out=transfer_out_player)
                
                if new_team.check_team_legality():
                    player_pool.append(transfer_out_player)
                    player_pool.remove(transfer_in_player)
                    current_team = new_team

                    if not current_team.check_team_legality():
                        raise ValueError("Current team is not legal")
            
                else:
                    continue

                n_transfers = count_number_of_transfers(current_team, self.team)
                
                if (total_points := calculate_total_team_points(current_team, n_free_transfers, n_transfers)) > best_team_points:
                    best_team_points = total_points
                    best_team = current_team
        
        self.team = best_team

