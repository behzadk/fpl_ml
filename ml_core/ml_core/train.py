from typing import Callable, Optional, Union

import mlflow
from sklearn.base import ClassifierMixin, RegressorMixin
from torchmetrics import Metric

from ml_core.sklearn_lightning_wrapper import SklearnModel
from ml_core.torch_lightning_wrapper import BasicTorchModel
from ml_core.utils import set_random_seeds, get_base_class
import torch
from functools import partial
import lightning.pytorch as pl
import torchmetrics
from ml_core.user import User
from ml_core.preprocessing import DataframePipeline



def run_train(
    user: User,
    dataset: partial[torch.utils.data.Dataset],
    datamodule: partial[pl.LightningDataModule],
    model: Union[RegressorMixin, torch.nn.Module],
    preprocessing: DataframePipeline,
    metrics: dict,
    visualisations: dict,
    optimizer: Optional[partial[torch.optim.Optimizer]],
    loss: Optional[Union[torch.nn.Module, torchmetrics.Metric]] = None,
    trainer: Optional[pl.Trainer] = None

):
    """Main train task function, This takes all the configs and uses them to train an sklearn model.

    TODO:
        Add support for pytorch training

    Args:
        user: Provides information defining mlflow runs dir, device type and random seed.
        dataset: Partially instantiated dataset, used to define how batches are generated.
        datamodule: Partially instantiated data_module, defines how we setup our data and manage dataloaders.
        model: Model we want train.
        preprocessing: Pipeline of `DataframeProcessingStep`'s.
        metrics: Dictionary of metrics to use for qunantifying our model in training and validation.
        visualisations: Dictionary of visualisations we want to apply to our data.
    """

    # Set mlflow tracking uri
    tracking_uri = f"file://{user.mlruns_dir}"
    mlflow.end_run()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name="Test")

    model_base_class = get_base_class(model)
    datamodule = datamodule()
    # datamodule.setup("fit")

    # # Get data dimensions
    # data_input_dim = datamodule.train_dataloader().dataset.get_input_dim()
    # data_output_dim = datamodule.train_dataloader().dataset.get_output_dim()

    # logger.info(f"Data input dim: {data_input_dim}, data output dim: {data_output_dim}")

    if model_base_class == "sklearn":
        output_metrics = train_sklearn(
            datamodule,
            model,
            metrics=metrics,
            experiment_name="Test",
            visualisations=visualisations,
            return_validation_metrics=None,
            random_seed=user.random_seed,
        )

    elif model_base_class == "lightning" or model_base_class=='torch' or model_base_class == "ml_core":
        output_metrics = train_torch(
            data_module=datamodule,
            model=model,
            metrics=metrics,
            partial_optimizer=optimizer,
            loss=loss,
            trainer=trainer,
            experiment_name="Test",
            visualisations=visualisations,
            return_validation_metrics=['val_MSE'],
            random_seed=user.random_seed,
        )

    # Pseudorandom so mlflow picks a new name for the next run
    set_random_seeds(None)

    return output_metrics

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
        mlflow.sklearn.autolog(log_datasets=False)

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

def train_torch(data_module: pl.LightningDataModule,
    model: pl.LightningModule,
    metrics: dict[str, Metric],
    experiment_name: str,
    visualisations: dict[str, Callable],
    partial_optimizer: partial[torch.optim.Optimizer],
    trainer: pl.Trainer,
    loss: Union[torch.nn.Module, torchmetrics.Metric],
    return_validation_metrics: Optional[list[str]] = None,
    random_seed: Union[int, None] = None, 
    ):

    mlflow.set_experiment(experiment_name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)


    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model = BasicTorchModel(model, visualisations=visualisations, metrics=metrics)

        mlflow.pytorch.autolog(log_datasets=False)
        
        # Seed everything and log the seed
        random_seed = set_random_seeds(random_seed)
        mlflow.log_param("seed", random_seed)

        mlflow.pyfunc.log_model(
            "preprocessing_pipeline", python_model=data_module.preprocessing_pipeline
        )

        mlflow.log_param("batch_size", data_module.batch_size)
        model.setup_train(data_module=data_module, metrics=metrics, partial_optimizer=partial_optimizer, loss=loss)

        # This setup is only run at the first instance of training
        trainer.fit(model=model, train_dataloaders=data_module)
        validation_outputs = trainer.validate(model=model, dataloaders=data_module)
        outputs = validation_outputs[0]


        trainer.test(model=model, dataloaders=data_module)

        
        # After training, log figures for training and validation steps
        with torch.no_grad():
            model._log_metrics(
                data_module.train_dataloader().dataset[:], stage="training", log_visualisations=True, log_metrics=False
            )
            model._log_metrics(
                data_module.val_dataloader().dataset[:], stage="val", log_visualisations=True, log_metrics=False
            )
            model._log_metrics(
                data_module.test_dataloader().dataset[:], stage="test", log_visualisations=True, log_metrics=False
            )

        if return_validation_metrics:
            output_metrics = tuple(outputs[x] for x in return_validation_metrics)
            
            if len(output_metrics) == 1:
                return output_metrics[0]

            else:
                return output_metrics