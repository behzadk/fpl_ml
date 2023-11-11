class CategoricalFeatures:
    TEAM_NAME = "X_team_name"
    WAS_HOME = "X_was_home"
    OPPONENT_TEAM = "X_opponent_team"
    ELEMENT_TYPE = "X_element_type"
    ALL = [TEAM_NAME, WAS_HOME, OPPONENT_TEAM, ELEMENT_TYPE]


class NumericalFeatures:
    GAME_WEEK = "X_game_week"
    VALUE = "X_value"

    ALL = [GAME_WEEK, VALUE]


class Prefixes:
    X = "X_"
    ROLLING_AVERAGE = "X_rolling_"


class Labels:
    TOTAL_POINTS = "total_points"
