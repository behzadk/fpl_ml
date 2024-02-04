from fpl_ml_projects.fpl_ml.simulation.agent import BaseAgent, HillClimbingAgent
from fpl_ml_projects.fpl_ml.constants import NumericalFeatures, Labels
from fpl_ml_projects.fpl_ml.simulation.environment import FPLEnv
from fpl_ml_projects.fpl_ml.constants import RewardType
import pandas as pd
import plotly.express as px

from ml_core.mlflow_utils import get_experiment_runs, load_model
from ml_core.data_module import DataModuleLoadedFromCSV
import mlflow

data_dir = "/mnt/c/Users/behza/Documents/code/data/fpl_ml/processed/"
test_data_path = data_dir + "test_data.csv"
tracking_uri = "file:///mnt/c/Users/behza/Documents/code/data/fpl_ml/mlruns"
experiment_name = "Test"


def load_preprocessing_pipeline(run_id, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(
        f"runs:/{run_id}/preprocessing_pipeline/"
    ).unwrap_python_model()
    return model


mlflow_runs_df = get_experiment_runs(
    experiment_name=experiment_name, tracking_uri=tracking_uri
)
mlflow_runs_df.sort_values(
    by=["metrics.val_SpearmanCorrCoef"], inplace=True, ascending=False
)
best_run_id = mlflow_runs_df.iloc[0]["run_id"]
model = load_model(run_id=best_run_id, tracking_uri=tracking_uri)
preprocessing_pipeline = load_preprocessing_pipeline(
    run_id=best_run_id, tracking_uri=tracking_uri
)


# Read test data
test_df = pd.read_csv(test_data_path)

# Transform test_data
processed_test_data = preprocessing_pipeline.transform(test_df)

pred = model.predict(processed_test_data["X"])

test_df[Labels.PREDICTED_TOTAL_POINTS] = model.predict(processed_test_data["X"])

season_df = test_df


agent_1 = BaseAgent(name="base_agent_1", team=[])
agent_2 = HillClimbingAgent(
    name="hill_agent_true_data",
    team=[],
    n_iterations=500,
    n_restarts=10,
    points_column=Labels.TOTAL_POINTS,
)
agent_3 = HillClimbingAgent(
    name="hill_agent_pred_data",
    team=[],
    n_iterations=500,
    n_restarts=10,
    points_column=Labels.PREDICTED_TOTAL_POINTS,
)

gw_1 = season_df[season_df[NumericalFeatures.GAME_WEEK] == 1]

agent_1.init_team(gw_1)
agent_2.init_team(gw_1)
agent_3.init_team(gw_1)

fpl_env = FPLEnv(
    agents=[agent_3],
    season_df=season_df,
    reward_type=RewardType.TOTAL_TEAM_POINTS,
)

fpl_env.run_simulation()
