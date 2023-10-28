import lightning.pytorch as pl
from typing import Optional, Union, Callable
from sklearn.base import ClassifierMixin, RegressorMixin
from torchmetrics import Metric
import mlflow
from fpl_ml.utils import set_random_seeds
from fpl_ml.sklearn_lightning_wrapper import SklearnModel


def train_sklearn(
    data_module: pl.LightningDataModule,
    model: Union[ClassifierMixin, RegressorMixin],
    metrics: dict[str, Metric],
    experiment_name: str,
    visualisations: dict[str, Callable],
    return_validation_metrics: Optional[list[str]] = None,
    random_seed: Union[int, None] = None,
) -> Union[float, tuple, None]:
    mlflow.set_experiment(experiment_name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Initialise model
    model = SklearnModel(model, metrics, visualisations)


    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.sklearn.autolog()
        
        # Seed everything and log the seed
        random_seed = set_random_seeds(random_seed)
        mlflow.log_param("seed", random_seed)

        # Fit model
        output_metrics = model.fit(data_module)

    # Optionally return metrics for hyperparameter optimization
    if return_validation_metrics:
        ret_metrics = tuple(output_metrics[x] for x in return_validation_metrics)

        if len(ret_metrics) == 1:
            return ret_metrics[0]

        return ret_metrics
