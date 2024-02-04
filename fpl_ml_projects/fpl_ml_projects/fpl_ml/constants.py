class CategoricalFeatures:
    TEAM_NAME = "X_team_name"
    WAS_HOME = "X_was_home"
    OPPONENT_TEAM = "X_opponent_team"
    ELEMENT_TYPE = "X_element_type"
    ALL = [TEAM_NAME, WAS_HOME, OPPONENT_TEAM, ELEMENT_TYPE]


class NumericalFeatures:
    GAME_WEEK = "X_game_week"
    VALUE = "X_value"
    PREV_YEAR_TOTAL_POINTS = "X_previous_year_total_points"

    ALL = [GAME_WEEK, VALUE, PREV_YEAR_TOTAL_POINTS]


class Prefixes:
    X = "X_"
    ROLLING_AVERAGE = "X_rolling_"
    GAME_WEEK_AVERAGE = "X_gw_avg_"


class Labels:
    TOTAL_POINTS = "total_points"
    PREDICTED_TOTAL_POINTS = "pred_total_points"


class Positions:
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    FORWARD = 4
    ALL = [GOALKEEPER, DEFENDER, MIDFIELDER, FORWARD]


class RewardType:
    TOTAL_TEAM_POINTS = "total_team_points"
