"""
Tests for Deep Tree Echo State Network Reservoir P-System Framework
"""

import unittest
import numpy as np
from frappe.tests import IntegrationTestCase

from erpnext.dt_esn_rp import (
	DeepTreeESNRP,
	EchoStateNetworkReservoir,
	PSystemMembraneComputing,
	AffectiveAgency,
)


class TestDeepTreeESNRP(IntegrationTestCase):
	"""Test cases for the main DT-(ESN)-RP framework."""
	
	def test_framework_initialization(self):
		"""Test framework initialization."""
		framework = DeepTreeESNRP(
			reservoir_size=100,
			spectral_radius=0.9,
			membrane_layers=2,
		)
		
		self.assertIsNotNone(framework)
		self.assertEqual(framework.reservoir_size, 100)
		self.assertEqual(framework.membrane_layers, 2)
		self.assertIsNotNone(framework.esn_reservoir)
		self.assertIsNotNone(framework.p_system)
	
	def test_framework_process(self):
		"""Test processing data through framework."""
		framework = DeepTreeESNRP(reservoir_size=50, membrane_layers=2)
		
		# Create test input
		input_data = np.random.randn(10)
		
		# Process data
		result = framework.process(input_data)
		
		# Verify results
		self.assertIn("reservoir_output", result)
		self.assertIn("membrane_output", result)
		self.assertIn("affective_output", result)
		self.assertIn("integrated_state", result)
	
	def test_framework_training(self):
		"""Test framework training."""
		framework = DeepTreeESNRP(reservoir_size=50, membrane_layers=2)
		
		# Create training data
		input_data = np.random.randn(20, 10)
		target_data = np.random.randn(20, 5)
		
		# Train framework
		metrics = framework.train(input_data, target_data)
		
		# Verify training metrics
		self.assertIn("training_error", metrics)
		self.assertIn("readout_weights_norm", metrics)
		self.assertIsInstance(metrics["training_error"], float)
	
	def test_framework_prediction(self):
		"""Test framework prediction."""
		framework = DeepTreeESNRP(reservoir_size=50, membrane_layers=2)
		
		# Create training data
		input_data = np.random.randn(20, 10)
		target_data = np.random.randn(20, 5)
		
		# Train framework
		framework.train(input_data, target_data)
		
		# Make prediction
		test_input = np.random.randn(10)
		prediction = framework.predict(test_input)
		
		# Verify prediction
		self.assertIsInstance(prediction, np.ndarray)
		self.assertEqual(len(prediction), 5)
	
	def test_framework_info(self):
		"""Test getting framework information."""
		framework = DeepTreeESNRP(reservoir_size=100, membrane_layers=3)
		
		info = framework.get_framework_info()
		
		self.assertIn("reservoir_size", info)
		self.assertIn("spectral_radius", info)
		self.assertIn("membrane_layers", info)
		self.assertEqual(info["reservoir_size"], 100)
		self.assertEqual(info["membrane_layers"], 3)


class TestEchoStateNetworkReservoir(IntegrationTestCase):
	"""Test cases for Echo State Network Reservoir."""
	
	def test_reservoir_initialization(self):
		"""Test reservoir initialization."""
		reservoir = EchoStateNetworkReservoir(
			reservoir_size=100,
			spectral_radius=0.9,
		)
		
		self.assertEqual(reservoir.reservoir_size, 100)
		self.assertEqual(reservoir.spectral_radius, 0.9)
	
	def test_reservoir_compute(self):
		"""Test reservoir computation."""
		reservoir = EchoStateNetworkReservoir(reservoir_size=50)
		
		# Create test input
		input_data = np.random.randn(10)
		
		# Compute
		result = reservoir.compute(input_data)
		
		# Verify results
		self.assertIn("reservoir_state", result)
		self.assertIn("activation_energy", result)
		self.assertIn("activation_stats", result)
		self.assertEqual(len(result["reservoir_state"]), 50)
	
	def test_reservoir_reset(self):
		"""Test reservoir reset."""
		reservoir = EchoStateNetworkReservoir(reservoir_size=50)
		
		# Compute with input
		input_data = np.random.randn(10)
		reservoir.compute(input_data)
		
		# Verify state is not zero
		self.assertGreater(np.sum(np.abs(reservoir.reservoir_state)), 0)
		
		# Reset
		reservoir.reset()
		
		# Verify state is zero
		self.assertEqual(np.sum(np.abs(reservoir.reservoir_state)), 0)
	
	def test_reservoir_spectral_radius(self):
		"""Test reservoir spectral radius."""
		reservoir = EchoStateNetworkReservoir(
			reservoir_size=50,
			spectral_radius=0.95,
		)
		
		# Initialize weights
		input_data = np.random.randn(10)
		reservoir.compute(input_data)
		
		# Get info
		info = reservoir.get_reservoir_info()
		
		# Verify spectral radius is close to target
		self.assertAlmostEqual(
			info["actual_spectral_radius"],
			0.95,
			delta=0.1,
		)


class TestPSystemMembraneComputing(IntegrationTestCase):
	"""Test cases for P-System Membrane Computing."""
	
	def test_psystem_initialization(self):
		"""Test P-System initialization."""
		p_system = PSystemMembraneComputing(num_membranes=3)
		
		self.assertEqual(len(p_system.membranes), 3)
		self.assertEqual(p_system.num_membranes, 3)
	
	def test_psystem_evolution(self):
		"""Test membrane evolution."""
		p_system = PSystemMembraneComputing(num_membranes=2)
		
		# Create test input
		input_state = {"reservoir_state": np.random.randn(100)}
		
		# Evolve
		result = p_system.evolve(input_state)
		
		# Verify results
		self.assertIn("membrane_states", result)
		self.assertIn("system_energy", result)
		self.assertIn("active_membranes", result)
		self.assertEqual(len(result["membrane_states"]), 2)
	
	def test_psystem_info(self):
		"""Test getting P-System information."""
		p_system = PSystemMembraneComputing(num_membranes=3)
		
		info = p_system.get_membrane_info()
		
		self.assertIn("num_membranes", info)
		self.assertIn("active_membranes", info)
		self.assertIn("system_energy", info)
		self.assertEqual(info["num_membranes"], 3)


class TestAffectiveAgency(IntegrationTestCase):
	"""Test cases for Affective Agency."""
	
	def test_affective_initialization(self):
		"""Test affective agency initialization."""
		agency = AffectiveAgency()
		
		self.assertIsNotNone(agency)
		self.assertEqual(len(agency.emotion_states), 10)
	
	def test_emotion_processing(self):
		"""Test emotion processing."""
		agency = AffectiveAgency()
		
		# Create test states
		reservoir_state = {
			"reservoir_state": np.random.randn(100),
			"activation_energy": 50.0,
		}
		membrane_state = {
			"system_energy": 30.0,
		}
		
		# Process emotions
		result = agency.process_emotions(reservoir_state, membrane_state)
		
		# Verify results
		self.assertIn("emotion_states", result)
		self.assertIn("affective_resonance", result)
		self.assertIn("valence", result)
		self.assertIn("arousal", result)
	
	def test_emotion_summary(self):
		"""Test emotion summary."""
		agency = AffectiveAgency()
		
		# Update some emotions
		agency.emotion_states["joy"] = 0.8
		agency.emotion_states["interest"] = 0.6
		
		# Get summary
		summary = agency.get_emotion_summary()
		
		# Verify summary
		self.assertIn("dominant_emotion", summary)
		self.assertIn("valence", summary)
		self.assertIn("arousal", summary)
		self.assertEqual(summary["dominant_emotion"], "joy")


class TestRidgeRegression(IntegrationTestCase):
	"""Test cases for Butcher B-Series Ridge Regression."""
	
	def test_ridge_regression_closed_form(self):
		"""Test closed-form ridge regression."""
		from erpnext.dt_esn_rp.ridge_regression import ButcherBSeriesRidgeRegression
		
		ridge = ButcherBSeriesRidgeRegression(
			regularization=1e-6,
			use_gradient_descent=False,
		)
		
		# Create test data
		X = np.random.randn(50, 20)
		y = np.random.randn(50, 5)
		
		# Fit
		weights = ridge.fit(X, y)
		
		# Verify
		self.assertIsNotNone(weights)
		self.assertEqual(weights.shape, (20, 5))
	
	def test_ridge_regression_gradient_descent(self):
		"""Test gradient descent ridge regression."""
		from erpnext.dt_esn_rp.ridge_regression import ButcherBSeriesRidgeRegression
		
		ridge = ButcherBSeriesRidgeRegression(
			regularization=1e-6,
			use_gradient_descent=True,
			max_iterations=100,
		)
		
		# Create test data
		X = np.random.randn(50, 20)
		y = np.random.randn(50, 5)
		
		# Fit
		weights = ridge.fit(X, y)
		
		# Verify
		self.assertIsNotNone(weights)
		self.assertEqual(weights.shape, (20, 5))
		self.assertGreater(len(ridge.training_history), 0)


class TestJuliaRicciFlow(IntegrationTestCase):
	"""Test cases for Julia Ricci Flow."""
	
	def test_ricci_flow_initialization(self):
		"""Test Ricci flow initialization."""
		from erpnext.dt_esn_rp.julia_ricci_flow import JuliaDifferentialRicciFlow
		
		ricci_flow = JuliaDifferentialRicciFlow(dimension=3)
		
		self.assertEqual(ricci_flow.dimension, 3)
		self.assertEqual(ricci_flow.metric_tensor.shape, (3, 3))
	
	def test_ricci_flow_computation(self):
		"""Test Ricci flow computation."""
		from erpnext.dt_esn_rp.julia_ricci_flow import JuliaDifferentialRicciFlow
		
		ricci_flow = JuliaDifferentialRicciFlow(dimension=3)
		
		# Create test state
		state = np.random.randn(20)
		
		# Compute flow
		result = ricci_flow.compute_ricci_flow(state)
		
		# Verify results
		self.assertIn("evolved_state", result)
		self.assertIn("metric_tensor", result)
		self.assertIn("scalar_curvature", result)
	
	def test_ricci_flow_geometric_info(self):
		"""Test geometric information."""
		from erpnext.dt_esn_rp.julia_ricci_flow import JuliaDifferentialRicciFlow
		
		ricci_flow = JuliaDifferentialRicciFlow(dimension=3)
		
		info = ricci_flow.get_geometric_info()
		
		self.assertIn("dimension", info)
		self.assertIn("scalar_curvature", info)
		self.assertIn("metric_eigenvalues", info)
		self.assertEqual(info["dimension"], 3)


class TestConfiguration(IntegrationTestCase):
	"""Test cases for configuration and integration."""
	
	def test_get_framework_config(self):
		"""Test getting framework configuration."""
		from erpnext.dt_esn_rp.config import get_framework_config
		
		config = get_framework_config()
		
		self.assertIn("reservoir", config)
		self.assertIn("p_system", config)
		self.assertIn("affective_agency", config)
		self.assertIn("ricci_flow", config)
	
	def test_initialize_framework(self):
		"""Test framework initialization from config."""
		from erpnext.dt_esn_rp.config import initialize_framework
		
		framework = initialize_framework()
		
		self.assertIsNotNone(framework)
		self.assertIsInstance(framework, DeepTreeESNRP)
	
	def test_framework_status(self):
		"""Test getting framework status."""
		from erpnext.dt_esn_rp.config import get_framework_status
		
		status = get_framework_status()
		
		self.assertIn("enabled", status)
		self.assertIn("version", status)
		self.assertIn("configuration", status)
		self.assertIn("components", status)
		self.assertTrue(status["enabled"])
