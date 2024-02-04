import unittest
from fpl_ml_projects.fpl_ml.simulation.agent import BaseAgent
from fpl_ml_projects.fpl_ml.constants import NumericalFeatures
import pkg_resources
from fpl_ml_projects.fpl_ml.simulation.environment import FPLEnv
import pandas as pd
import os


_TEST_DIR = pkg_resources.resource_filename(__name__, "test_data")


class TestFPLEnv(unittest.TestCase):
    def setUp(self):
        self.season_df = pd.read_csv(os.path.join(_TEST_DIR, "test_season.csv"))

        # Setup agents
        self.agent_n = []
        for i in range(2):
            base_agent = BaseAgent(name=f"test_agent_{i}", team=[])
            base_agent.init_team(
                self.season_df[self.season_df[NumericalFeatures.GAME_WEEK] == 1]
            )
            self.agent_n.append(base_agent)

        self.env = FPLEnv(self.agent_n, self.season_df)

    def test_step(self):
        obs, reward_n, done = self.env.step(
            teams=[agent.team for agent in self.agent_n]
        )
        print(reward_n, done)

    def test_get_observation_space(self):
        obs = self.env.get_observation_space(1)
        self.assertEqual(len(obs), 11)


if __name__ == "__main__":
    unittest.main()
