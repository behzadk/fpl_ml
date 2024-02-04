import unittest
from fpl_ml_projects.fpl_ml.simulation.player import Player
from fpl_ml_projects.fpl_ml.simulation.user import UserTeam

from fpl_ml_projects.fpl_ml.constants import Positions
import uuid
import pkg_resources

_TEST_DIR = pkg_resources.resource_filename(__name__, "test_data")


class TestUserTeam(unittest.TestCase):
    def setUp(self):
        self.goalkeepers = []
        self.defenders = []
        self.midfielders = []
        self.forwards = []

        # Make two goalkeepers
        for i in range(2):
            self.goalkeepers.append(
                Player(
                    f"Player_GK_{i}",
                    position=Positions.GOALKEEPER,
                    value=5.0,
                    club=str(uuid.uuid4()),
                )
            )

        # Make five defenders
        for i in range(5):
            self.defenders.append(
                Player(
                    f"Player_DEF_{i}",
                    position=Positions.DEFENDER,
                    value=5.0,
                    club=str(uuid.uuid4()),
                )
            )

        # Make five midfielders
        for i in range(5):
            self.midfielders.append(
                Player(
                    f"Player_MID_{i}",
                    position=Positions.MIDFIELDER,
                    value=5.0,
                    club=str(uuid.uuid4()),
                )
            )

        # Make three forwards
        for i in range(3):
            self.forwards.append(
                Player(
                    f"Player_FWD_{i}",
                    position=Positions.FORWARD,
                    value=5.0,
                    club=str(uuid.uuid4()),
                )
            )

        self.team = UserTeam(
            [*self.goalkeepers, *self.defenders, *self.midfielders, *self.forwards]
        )

    def test_transfer_player(self):
        player_in = Player(
            "Alex Brown",
            position=Positions.FORWARD,
            value=7,
            club="Chelsea",
        )
        player_out = self.forwards[0]

        self.team.transfer_player(player_in, player_out)

        self.assertNotIn(player_out, self.team.players)
        self.assertIn(player_in, self.team.players)

    def test_team_clone(self):
        team_clone = self.team.clone()

        self.assertEqual(team_clone.players, self.team.players)
        self.assertEqual(team_clone.players, self.team.players)

        self.assertIsNot(team_clone.players, self.team.players)

        player_in = Player(
            "Alex Brown",
            position=Positions.FORWARD,
            value=7,
            club="Chelsea",
        )

        team_clone.transfer_player(player_in, player_out=self.forwards[0])
        self.assertNotEqual(team_clone.players, self.team.players)


    def test_team_copy(self):
        team_copy = self.team.copy()

        self.assertIsNot(team_copy.players, self.team.players)

    def test_check_team_legality(self):
        team = self.team.copy()

        # Check that the team is legal
        self.assertTrue(team.check_team_legality())

        # Remove a player to make the team illegal
        team.players.remove(team.players[0])
        self.assertFalse(team.check_team_legality())

        # Add a player to make the team illegal
        team = self.team.copy()
        team.players.append(
            Player("Alex Brown", position=Positions.FORWARD, value=7, club="Chelsea")
        )
        self.assertFalse(team.check_team_legality())

        # Make all midfielders from the same club
        team = self.team.copy()
        for p in team.players[:4]:
            p.club = "Chelsea"
        self.assertFalse(team.check_team_legality())

        # Make illegal value of team
        team = self.team.copy()
        team.players[0].value = 1000
        self.assertFalse(team.check_team_legality())

        team = self.team.copy()
        self.assertTrue(team.check_team_legality())


if __name__ == "__main__":
    unittest.main()
