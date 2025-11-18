import numpy as np
from typing import Optional


class Adaline:
    """
    Adaptive Linear Neuron model trained with gradient descent to minimize mean squared error.
    """

    def __init__(
        self,
        learning_rate: float = 0.0001,
        epochs: int = 30,
        random_state: Optional[int] = None,
        threshold: float = 0.0,
    ) -> None:
        """
        Initializes the Adaline model with the given hyperparameters.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.threshold = threshold
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initializes model parameters using a small random normal distribution.
        """
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the net input of the model as a linear combination of features and weights.
        """
        if self.weights is None:
            raise ValueError("Model parameters are not initialized. Call fit before net_input.")
        return X @ self.weights + self.bias_

    @property
    def weights(self) -> np.ndarray:
        """
        Returns the weight vector, raising an error if the model has not been fitted.
        """
        if self.weights_ is None:
            raise ValueError("Model parameters are not initialized. Call fit first.")
        return self.weights_

    def activation(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the activation function, which for Adaline is the identity of the net input.
        """
        return self.net_input(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the continuous output scores before thresholding.
        """
        return self.activation(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Adaline":
        """
        Trains the Adaline model using batch gradient descent to minimize mean squared error.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        y = y.astype(float)

        for _ in range(self.epochs):
            outputs = self.net_input(X)
            errors = y - outputs
            self.weights_ += (self.learning_rate * X.T @ errors) / n_samples
            self.bias_ += self.learning_rate * errors.mean()
            loss = np.mean(errors**2)
            self.loss_history_.append(loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts binary class labels based on the continuous output and the configured threshold.
        """
        scores = self.decision_function(X)
        return np.where(scores >= self.threshold, 1, 0)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the continuous output scores of the model for analysis or custom thresholding.
        """
        return self.decision_function(X)
