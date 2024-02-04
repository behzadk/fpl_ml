from fpl_ml_projects.fpl_ml import constants
from fpl_ml_projects.fpl_ml.simulation.player import Player


class UserTeam:
    def __init__(self, players=list[Player]):
        self.players = players

    def transfer_player(self, player_in: Player, player_out: Player):
        self.players.remove(player_out)
        self.players.append(player_in)

    def check_team_legality(self) -> bool:
        positions_legal = self._check_legality_of_player_positions()
        team_value_legal = self._check_legality_of_team_value()
        players_from_same_club_legal = (
            self._check_legality_of_team_players_from_same_club()
        )

        if positions_legal and team_value_legal and players_from_same_club_legal:
            return True

        else:
            return False

    def _check_legality_of_player_positions(self) -> bool:
        d = self.count_players_in_each_position()

        for position in constants.Positions.ALL:
            players = d[position]

            if position == constants.Positions.GOALKEEPER:
                if players != 2:
                    return False

            if position == constants.Positions.DEFENDER:
                if players != 5:
                    return False

            if position == constants.Positions.MIDFIELDER:
                if players != 5:
                    return False

            if position == constants.Positions.FORWARD:
                if players != 3:
                    return False

        if len(self.players) != 15:
            return False

        return True

    def _check_legality_of_team_value(self) -> bool:
        if self.count_total_value_of_team() > 1000:
            return False

        return True

    def _check_legality_of_team_players_from_same_club(self) -> bool:
        d = self.count_players_from_each_club()

        for club in d:
            if d[club] > 3:
                return False

        return True

    def count_total_value_of_team(self) -> float:
        total_value = 0.0
        for player in self.players:
            total_value += player.value
        return total_value

    def count_players_in_each_position(self) -> dict:
        position_count = {}
        for player in self.players:
            if player.position in position_count:
                position_count[player.position] += 1
            else:
                position_count[player.position] = 1
        return position_count

    def count_players_from_each_club(self) -> dict:
        club_count = {}
        for player in self.players:
            if player.club in club_count:
                club_count[player.club] += 1
            else:
                club_count[player.club] = 1

        return club_count

    def get_player_names(self) -> list[str]:
        return [p.name for p in self.players]

    def clone(self):
        return type(self)(self.players[:])

    def copy(self):
        return type(self)([p.copy() for p in self.players])

    def __str__(self):
        return self.players
