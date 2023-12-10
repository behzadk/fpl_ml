from typing import Callable, List, Type, Union

import mlflow
import torch.nn as nn
from torch.nn.modules import Module

import ml_core



def _add_hidden_layers_sequentially(
    sequential_layers: List,
    hidden_layer_sizes: List[int],
    activation_function: Type[Module],
    dropout: float,
):
    """
    Adds hidden layers to a sequential layers list.

    Args:
        sequential_layers (List): A list of sequential layers.
        hidden_layer_sizes (List[int]): A list of hidden layer sizes.
        activation_function (Type[Module]): The activation function to use.
        dropout (float): The dropout rate to use.

    Returns:
        None
    """
    # Add hidden layers to a sequential layers list
    for h_idx in range(1, len(hidden_layer_sizes)):
        dim_in = hidden_layer_sizes[h_idx - 1]
        dim_out = hidden_layer_sizes[h_idx]

        sequential_layers.append(nn.Linear(in_features=dim_in, out_features=dim_out))
        sequential_layers.append(activation_function())
        sequential_layers.append(nn.Dropout(p=dropout))

        # Log hidden layer sizes
        mlflow.log_param(f"h{h_idx}_dim", hidden_layer_sizes[h_idx])


def neural_network_inferred_in_out_dimensions(
    dataset: Type[ml_core.dataset.Dataset],
    hidden_layer_sizes: List[int] = [2],
    activation_function: Type[Module] = nn.ReLU,
    dropout: float = 0.1,
    output_function: Union[None, Callable] = None,
) -> nn.Sequential:
    """
    Creates a PyTorch neural network model with inferred input and output dimensions.

    Args:
        dataset (Type[ml_core.dataset.Dataset]): The dataset to use for inferring input and output dimensions.
        hidden_layer_sizes (List[int], optional): A list of integers representing the number of neurons in each hidden layer. Defaults to [2].
        activation_function (Type[Module], optional): The activation function to use for each hidden layer. Defaults to nn.ReLU.
        dropout (float, optional): The dropout rate to use for each hidden layer. Defaults to 0.1.
        output_function (Union[None, Callable], optional): The output function to use for the final layer. Defaults to None.

    Returns:
        nn.Sequential: The PyTorch neural network model.
    """
    
    input_dim = dataset.get_input_dim()
    output_dim = dataset.get_output_dim()

    # Log dropout
    mlflow.log_param("dropout", dropout)

    # Make input layer
    sequential_layers = [
        nn.Linear(in_features=input_dim, out_features=hidden_layer_sizes[0]),
        activation_function,
        nn.Dropout(p=dropout),
    ]

    _add_hidden_layers_sequentially(
        sequential_layers,
        hidden_layer_sizes,
        dropout=dropout,
        activation_function=activation_function,
    )

    # Add final layer
    sequential_layers.append(
        nn.Linear(in_features=hidden_layer_sizes[-1], out_features=output_dim)
    )

    # Optionally add output function
    if output_function:
        sequential_layers.append(output_function)

    # Assemble into pytorch model
    model = nn.Sequential(*sequential_layers)

    return model



def neural_network(
    input_dim: int = 10,
    output_dim: int = 1,
    hidden_layer_sizes: List[int] = [5],
    activation_function: nn.Module = nn.ReLU,
    dropout: float = 0.8,
    output_function: Union[None, Callable] = None,
) -> nn.modules.container.Sequential:
    """
    Creates a neural network model with the given input and output dimensions, hidden layer sizes,
    activation function, dropout rate, and output function (if any).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        hidden_layer_sizes (List[int]): A list of integers representing the number of neurons in each hidden layer.
        activation_function (nn.Module): The activation function to use for each hidden layer.
        dropout (float): The dropout rate to use for each hidden layer.
        output_function (Union[None, Callable]): The output function to use for the final layer (if any).

    Returns:
        nn.modules.container.Sequential: A PyTorch sequential model object.
    """
    
    # Make input layer
    sequential_layers = [
        nn.Linear(in_features=input_dim, out_features=hidden_layer_sizes[0]),
        activation_function(),
        nn.Dropout(p=dropout),
    ]

    # Add hidden layers
    _add_hidden_layers_sequentially(
        sequential_layers,
        hidden_layer_sizes,
        dropout=dropout,
        activation_function=activation_function,
    )

    # Log hidden layer sizes
    for h_idx, h in enumerate(hidden_layer_sizes):
        mlflow.log_param(f"h{h_idx}_dim", h)

    # Log dropout
    mlflow.log_param("dropout", dropout)

    # Add final layer
    sequential_layers.append(
        nn.Linear(in_features=hidden_layer_sizes[-1], out_features=output_dim)
    )

    # Optinoanlly add output function
    if output_function:
        sequential_layers.append(output_function)

    # Assemble into pytorch model
    model = nn.Sequential(*sequential_layers)

    return model
