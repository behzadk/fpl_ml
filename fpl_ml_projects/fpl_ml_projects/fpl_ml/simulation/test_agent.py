import unittest
from fpl_ml_projects.fpl_ml.simulation.agent import BaseAgent
import pkg_resources
import pandas as pd
from fpl_ml_projects.fpl_ml.constants import NumericalFeatures

_TEST_DIR = pkg_resources.resource_filename(__name__, "test_data")


class TestBaseAgent(unittest.TestCase):
    def test_init_team(self):
        agent = BaseAgent(name="test_agent", team=[])
        season_df = pd.read_csv(f"{_TEST_DIR}/test_season.csv")
        season_df = season_df[season_df[NumericalFeatures.GAME_WEEK] == 1]

        agent.init_team(season_df)

        self.assertTrue(agent.team.check_team_legality())

if __name__ == "__main__":
    unittest.main()
