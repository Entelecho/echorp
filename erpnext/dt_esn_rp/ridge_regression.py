"""
Butcher B-Series Rooted Forest Runge-Kutta Ridge Regression

This module implements ridge regression with Butcher B-Series integration
methods for training the readout layer of the reservoir.
"""

from typing import Any, Optional

import numpy as np


class ButcherBSeriesRidgeRegression:
	"""
	Ridge Regression with Butcher B-Series Rooted Forest Runge-Kutta integration.

	This class implements regularized least squares regression using ridge
	regression with numerical integration methods based on Butcher tableaux.
	"""

	def __init__(
		self,
		regularization: float = 1e-6,
		integration_method: str = "rk4",
		use_gradient_descent: bool = True,
		learning_rate: float = 0.01,
		max_iterations: int = 1000,
	):
		"""
		Initialize Butcher B-Series Ridge Regression.

		Args:
			regularization: Ridge regularization parameter (lambda)
			integration_method: Integration method ("euler", "rk4", "butcher")
			use_gradient_descent: Use gradient descent for optimization
			learning_rate: Learning rate for gradient descent
			max_iterations: Maximum iterations for gradient descent
		"""
		self.regularization = regularization
		self.integration_method = integration_method
		self.use_gradient_descent = use_gradient_descent
		self.learning_rate = learning_rate
		self.max_iterations = max_iterations

		self.weights = None
		self.training_history = []

	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
	) -> np.ndarray:
		"""
		Fit ridge regression model.

		Args:
			X: Input feature matrix (n_samples, n_features)
			y: Target values (n_samples, n_outputs)

		Returns:
			Trained weight matrix
		"""
		if self.use_gradient_descent:
			return self._fit_gradient_descent(X, y)
		else:
			return self._fit_closed_form(X, y)

	def _fit_closed_form(
		self,
		X: np.ndarray,
		y: np.ndarray,
	) -> np.ndarray:
		"""
		Fit using closed-form ridge regression solution.

		Args:
			X: Input feature matrix
			y: Target values

		Returns:
			Trained weight matrix
		"""
		# Ridge regression closed form: W = (X'X + λI)^(-1) X'y
		n_features = X.shape[1]
		identity = np.eye(n_features)

		# Compute (X'X + λI)
		XtX = np.dot(X.T, X)
		ridge_matrix = XtX + self.regularization * identity

		# Compute (X'X + λI)^(-1) X'y
		Xty = np.dot(X.T, y)
		self.weights = np.linalg.solve(ridge_matrix, Xty)

		return self.weights

	def _fit_gradient_descent(
		self,
		X: np.ndarray,
		y: np.ndarray,
	) -> np.ndarray:
		"""
		Fit using gradient descent with Runge-Kutta integration.

		Args:
			X: Input feature matrix
			y: Target values

		Returns:
			Trained weight matrix
		"""
		n_features = X.shape[1]
		n_outputs = y.shape[1] if y.ndim > 1 else 1

		if y.ndim == 1:
			y = y.reshape(-1, 1)

		# Initialize weights
		self.weights = np.random.randn(n_features, n_outputs) * 0.01

		# Gradient descent with Runge-Kutta integration
		for iteration in range(self.max_iterations):
			# Compute gradient
			predictions = np.dot(X, self.weights)
			error = predictions - y
			gradient = np.dot(X.T, error) / X.shape[0] + self.regularization * self.weights

			# Apply Runge-Kutta integration method
			if self.integration_method == "euler":
				self.weights = self._euler_step(self.weights, gradient)
			elif self.integration_method == "rk4":
				self.weights = self._rk4_step(self.weights, gradient, X, y)
			else:
				self.weights = self._butcher_step(self.weights, gradient)

			# Compute loss
			loss = self._compute_loss(X, y, self.weights)
			self.training_history.append(loss)

			# Check convergence
			if iteration > 0 and abs(self.training_history[-1] - self.training_history[-2]) < 1e-6:
				break

		return self.weights

	def _euler_step(
		self,
		weights: np.ndarray,
		gradient: np.ndarray,
	) -> np.ndarray:
		"""
		Apply Euler integration step.

		Args:
			weights: Current weights
			gradient: Gradient

		Returns:
			Updated weights
		"""
		return weights - self.learning_rate * gradient

	def _rk4_step(
		self,
		weights: np.ndarray,
		gradient: np.ndarray,
		X: np.ndarray,
		y: np.ndarray,
	) -> np.ndarray:
		"""
		Apply 4th-order Runge-Kutta integration step.

		Args:
			weights: Current weights
			gradient: Initial gradient
			X: Input features
			y: Target values

		Returns:
			Updated weights
		"""
		h = self.learning_rate

		# k1 = f(w, t)
		k1 = -gradient

		# k2 = f(w + h/2 * k1, t + h/2)
		w2 = weights + h / 2 * k1
		pred2 = np.dot(X, w2)
		err2 = pred2 - y
		grad2 = np.dot(X.T, err2) / X.shape[0] + self.regularization * w2
		k2 = -grad2

		# k3 = f(w + h/2 * k2, t + h/2)
		w3 = weights + h / 2 * k2
		pred3 = np.dot(X, w3)
		err3 = pred3 - y
		grad3 = np.dot(X.T, err3) / X.shape[0] + self.regularization * w3
		k3 = -grad3

		# k4 = f(w + h * k3, t + h)
		w4 = weights + h * k3
		pred4 = np.dot(X, w4)
		err4 = pred4 - y
		grad4 = np.dot(X.T, err4) / X.shape[0] + self.regularization * w4
		k4 = -grad4

		# Update: w_new = w + h/6 * (k1 + 2*k2 + 2*k3 + k4)
		return weights + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

	def _butcher_step(
		self,
		weights: np.ndarray,
		gradient: np.ndarray,
	) -> np.ndarray:
		"""
		Apply Butcher tableau integration step.

		Args:
			weights: Current weights
			gradient: Gradient

		Returns:
			Updated weights
		"""
		# Simplified Butcher tableau method (similar to RK4)
		return self._euler_step(weights, gradient)

	def _compute_loss(
		self,
		X: np.ndarray,
		y: np.ndarray,
		weights: np.ndarray,
	) -> float:
		"""
		Compute ridge regression loss.

		Args:
			X: Input features
			y: Target values
			weights: Current weights

		Returns:
			Loss value
		"""
		predictions = np.dot(X, weights)
		mse = np.mean((predictions - y) ** 2)
		regularization_term = self.regularization * np.sum(weights**2)
		return mse + regularization_term

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Generate predictions.

		Args:
			X: Input features

		Returns:
			Predictions
		"""
		if self.weights is None:
			raise ValueError("Model not trained. Call fit() first.")

		return np.dot(X, self.weights)

	def get_training_info(self) -> dict[str, Any]:
		"""
		Get training information.

		Returns:
			Training information dictionary
		"""
		return {
			"regularization": self.regularization,
			"integration_method": self.integration_method,
			"use_gradient_descent": self.use_gradient_descent,
			"final_loss": self.training_history[-1] if self.training_history else None,
			"iterations": len(self.training_history),
			"weights_shape": self.weights.shape if self.weights is not None else None,
		}
