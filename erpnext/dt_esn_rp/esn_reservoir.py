"""
Echo State Network Reservoir Computing Implementation

This module implements the Echo State Network (ESN) reservoir for temporal
information processing with dynamic memory and non-linear transformations.
"""

from typing import Any, Optional

import numpy as np


class EchoStateNetworkReservoir:
	"""
	Echo State Network Reservoir for temporal computing.

	The ESN provides a dynamic reservoir of recurrently connected neurons
	that can capture temporal dependencies in input signals.
	"""

	def __init__(
		self,
		reservoir_size: int = 1000,
		spectral_radius: float = 0.9,
		input_scaling: float = 1.0,
		leak_rate: float = 0.3,
		connectivity: float = 0.1,
		random_seed: int | None = None,
	):
		"""
		Initialize the Echo State Network Reservoir.

		Args:
			reservoir_size: Number of neurons in the reservoir
			spectral_radius: Spectral radius for echo state property
			input_scaling: Scaling factor for input weights
			leak_rate: Leaking rate for reservoir dynamics
			connectivity: Sparsity of reservoir connections
			random_seed: Random seed for reproducibility
		"""
		self.reservoir_size = reservoir_size
		self.spectral_radius = spectral_radius
		self.input_scaling = input_scaling
		self.leak_rate = leak_rate
		self.connectivity = connectivity

		if random_seed is not None:
			np.random.seed(random_seed)

		# Initialize reservoir state
		self.reservoir_state = np.zeros(reservoir_size)

		# Initialize reservoir weights (will be set on first input)
		self.W_reservoir = None
		self.W_input = None
		self.input_dim = None

	def _initialize_weights(self, input_dim: int):
		"""
		Initialize reservoir and input weights.

		Args:
			input_dim: Dimensionality of input data
		"""
		self.input_dim = input_dim

		# Initialize sparse reservoir weights
		W = np.random.randn(self.reservoir_size, self.reservoir_size)
		mask = np.random.rand(self.reservoir_size, self.reservoir_size) < self.connectivity
		W = W * mask

		# Scale to desired spectral radius
		eigenvalues = np.linalg.eigvals(W)
		max_eigenvalue = np.max(np.abs(eigenvalues))
		self.W_reservoir = W * (self.spectral_radius / max_eigenvalue)

		# Initialize input weights
		self.W_input = np.random.randn(self.reservoir_size, input_dim) * self.input_scaling

	def compute(self, input_data: np.ndarray) -> dict[str, Any]:
		"""
		Compute reservoir activation for given input.

		Args:
			input_data: Input data array

		Returns:
			Dictionary containing reservoir state and activation
		"""
		# Ensure input is 1D array
		if input_data.ndim > 1:
			input_data = input_data.flatten()

		# Initialize weights on first call
		if self.W_reservoir is None:
			self._initialize_weights(len(input_data))

		# Compute reservoir update
		# x(t+1) = (1-a)x(t) + a tanh(W_res·x(t) + W_in·u(t))
		pre_activation = np.dot(self.W_reservoir, self.reservoir_state) + np.dot(self.W_input, input_data)

		self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + self.leak_rate * np.tanh(
			pre_activation
		)

		# Compute activation statistics
		activation_stats = self._compute_activation_stats()

		return {
			"reservoir_state": self.reservoir_state.copy(),
			"activation_energy": np.sum(self.reservoir_state**2),
			"activation_stats": activation_stats,
		}

	def _compute_activation_stats(self) -> dict[str, float]:
		"""
		Compute statistics of reservoir activation.

		Returns:
			Dictionary of activation statistics
		"""
		return {
			"mean": float(np.mean(self.reservoir_state)),
			"std": float(np.std(self.reservoir_state)),
			"min": float(np.min(self.reservoir_state)),
			"max": float(np.max(self.reservoir_state)),
			"sparsity": float(np.sum(np.abs(self.reservoir_state) < 0.1) / self.reservoir_size),
		}

	def reset(self):
		"""Reset reservoir state to zero."""
		self.reservoir_state = np.zeros(self.reservoir_size)

	def get_reservoir_info(self) -> dict[str, Any]:
		"""
		Get information about the reservoir configuration.

		Returns:
			Reservoir configuration information
		"""
		info = {
			"reservoir_size": self.reservoir_size,
			"spectral_radius": self.spectral_radius,
			"input_scaling": self.input_scaling,
			"leak_rate": self.leak_rate,
			"connectivity": self.connectivity,
		}

		if self.W_reservoir is not None:
			# Compute actual spectral radius
			eigenvalues = np.linalg.eigvals(self.W_reservoir)
			actual_spectral_radius = float(np.max(np.abs(eigenvalues)))
			info["actual_spectral_radius"] = actual_spectral_radius
			info["input_dim"] = self.input_dim

		return info
