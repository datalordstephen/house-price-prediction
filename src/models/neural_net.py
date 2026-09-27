"""Feed-forward neural network regressor (multi-layer perceptron)."""

from sklearn.neural_network import MLPRegressor

from src.config import RANDOM_STATE

NAME = "Neural Net (MLP)"


def build():
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
