# Fantasy Premier League Machine Learning
Repository for preparation, analysis, model fitting and prediction for fantasy premier league data.

Credit to https://github.com/vaastav/Fantasy-Premier-League for building a fantastic data resource.

### Project aims
First stage of the project is to train models and perform model selection, predicting player scores for each game week of a premier league season. We are only using features that are available on https://fantasy.premierleague.com. We take a model agnostic approach to the models we use, this repo will support scikit-learn and pytorch models. 

Second stage of the project is to implement heuristic algorithms that build teams within the game restrictions to maximise the expeceted points of a team for either short-term or long-term gains.

We aim to provide a platform for fast experimental iteration and model comparison using a combination of [pytorch-lightning](https://lightning.ai) for defining standardised datamodules and dataloaders, [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) for config management and boiler-plate free hyperparameter optimization with [Optuna](https://optuna.org) or [Hyperopt](http://hyperopt.github.io/hyperopt/). All experiments are logged in a standardised fashion using [MLflow](https://mlflow.org)

### Example output

Example below shows performance on validation set for predicting player scores for the `2019-2020` season. Each datapoint is the number of points a player got in a given game week. The purple line is the identity (`x == y`). Datapoints can come from any season. Future efforts will hold out a season for the validation set and a season for the test set. 
![image](https://github.com/behzadk/fpl_ml/assets/15074455/56fddcb2-5b92-499f-80a8-25e6e7190d03)


### Current features
- Data preparation
- Train sklearn models
- Logging with mlflow
- Hydra-zen configs

### TODO:
  - Script for downloading raw data
  - Unittesting
  - Pytorch support
  - Heueristic simulations for building teams
  - Containerisation
