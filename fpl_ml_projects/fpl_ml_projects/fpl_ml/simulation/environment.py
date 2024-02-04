from fpl_ml_projects.fpl_ml import constants
import pandas as pd
from fpl_ml_projects.fpl_ml.simulation.agent import BaseAgent
from fpl_ml_projects.fpl_ml.simulation.user import UserTeam
from loguru import logger


class FPLEnv:
    def __init__(
        self,
        agents: list[BaseAgent],
        season_df: pd.DataFrame,
        reward_type=constants.RewardType.TOTAL_TEAM_POINTS,
    ):
        self.gw = 1
        self.season_df = season_df

        self.reward_type = reward_type

        self.agent_n = agents
        self._max_gw = self.season_df[constants.NumericalFeatures.GAME_WEEK].max()

    def run_simulation(self):
        done = False

        self.agent_rewards = {"gw": [], "agent": [], "reward": []}

        for agent in self.agent_n:
            agent.init_team(self.get_observation_space(self.gw))

        while not done:
            logger.info(f"Game week: {self.gw}")

            observation, reward_n, done = self.step(
                teams=[agent.team for agent in self.agent_n]
            )

            for i, agent in enumerate(self.agent_n):
                logger.info(f"Agent: {agent.name}, Reward: {reward_n[i]}")

                self.agent_rewards["gw"].append(self.gw)
                self.agent_rewards["agent"].append(agent.name)
                self.agent_rewards["reward"].append(reward_n[i])

                if not done:
                    # Step the agent through the next observation
                    agent.step_team(observation, n_free_transfers=1)

    def get_observation_space(self, gw) -> tuple:
        gw_df = self.season_df[
            self.season_df[constants.NumericalFeatures.GAME_WEEK] == gw
        ]

        return gw_df

    def get_reward(self, team: UserTeam) -> float:
        if self.reward_type == constants.RewardType.TOTAL_TEAM_POINTS:
            return self.calculate_total_team_points_reward(team)
        else:
            raise NotImplementedError

    def calculate_total_team_points_reward(self, team: UserTeam) -> float:
        gw_df = self.get_observation_space(self.gw)

        reward = 0.0

        for player in team.players:
            player_df = gw_df[gw_df["player_name"] == player.name]

            if len(player_df) == 0:
                continue

            reward += player_df[constants.Labels.TOTAL_POINTS].values[0]

        return reward

    def step(self, teams: UserTeam) -> bool:
        """Perform a step in the environment

        Args:
            teams: list of teams

        Returns:
            observation: new observation
            reward_n: list of rewards
            done: boolean indicating if the season is over
        """
        reward_n = []
        for t in teams:
            reward_n.append(self.get_reward(t))

        # Increment game week
        self.gw += 1

        # Check if season is over
        done = False
        if self.gw > self._max_gw:
            done = True
            observation = None

        else:
            # Get new observation
            observation = self.get_observation_space(self.gw)

        return (observation, reward_n, done)
