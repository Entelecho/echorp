"""
Standalone tests for DT-(ESN)-RP framework (without frappe dependencies)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_esn_reservoir():
	"""Test Echo State Network Reservoir."""
	print("Testing ESN Reservoir...")
	from dt_esn_rp.esn_reservoir import EchoStateNetworkReservoir

	reservoir = EchoStateNetworkReservoir(reservoir_size=50, spectral_radius=0.9)

	# Test computation
	input_data = np.random.randn(10)
	result = reservoir.compute(input_data)

	assert "reservoir_state" in result
	assert "activation_energy" in result
	assert len(result["reservoir_state"]) == 50
	print("✓ ESN Reservoir test passed")


def test_p_system():
	"""Test P-System Membrane Computing."""
	print("Testing P-System...")
	from dt_esn_rp.p_system import PSystemMembraneComputing

	p_system = PSystemMembraneComputing(num_membranes=2)

	# Test evolution
	input_state = {"reservoir_state": np.random.randn(100)}
	result = p_system.evolve(input_state)

	assert "membrane_states" in result
	assert "system_energy" in result
	assert len(result["membrane_states"]) == 2
	print("✓ P-System test passed")


def test_affective_agency():
	"""Test Affective Agency."""
	print("Testing Affective Agency...")
	from dt_esn_rp.affective_agency import AffectiveAgency

	agency = AffectiveAgency()

	# Test emotion processing
	reservoir_state = {"reservoir_state": np.random.randn(100), "activation_energy": 50.0}
	membrane_state = {"system_energy": 30.0}

	result = agency.process_emotions(reservoir_state, membrane_state)

	assert "emotion_states" in result
	assert "affective_resonance" in result
	assert "valence" in result
	assert "arousal" in result
	print("✓ Affective Agency test passed")


def test_ridge_regression():
	"""Test Ridge Regression."""
	print("Testing Ridge Regression...")
	from dt_esn_rp.ridge_regression import ButcherBSeriesRidgeRegression

	ridge = ButcherBSeriesRidgeRegression(regularization=1e-6, use_gradient_descent=False)

	# Test fitting
	X = np.random.randn(50, 20)
	y = np.random.randn(50, 5)

	weights = ridge.fit(X, y)

	assert weights is not None
	assert weights.shape == (20, 5)
	print("✓ Ridge Regression test passed")


def test_julia_ricci_flow():
	"""Test Julia Ricci Flow."""
	print("Testing Julia Ricci Flow...")
	from dt_esn_rp.julia_ricci_flow import JuliaDifferentialRicciFlow

	ricci_flow = JuliaDifferentialRicciFlow(dimension=3)

	# Test flow computation
	state = np.random.randn(20)
	result = ricci_flow.compute_ricci_flow(state)

	assert "evolved_state" in result
	assert "metric_tensor" in result
	assert "scalar_curvature" in result
	print("✓ Julia Ricci Flow test passed")


def test_integration():
	"""Test integrated framework (without frappe)."""
	print("Testing Integration...")
	from dt_esn_rp.esn_reservoir import EchoStateNetworkReservoir
	from dt_esn_rp.p_system import PSystemMembraneComputing
	from dt_esn_rp.affective_agency import AffectiveAgency

	# Initialize components
	reservoir = EchoStateNetworkReservoir(reservoir_size=50)
	p_system = PSystemMembraneComputing(num_membranes=2)
	agency = AffectiveAgency()

	# Process data through pipeline
	input_data = np.random.randn(10)

	# Step 1: ESN
	reservoir_output = reservoir.compute(input_data)

	# Step 2: P-System
	membrane_output = p_system.evolve(reservoir_output)

	# Step 3: Affective Agency
	affective_output = agency.process_emotions(reservoir_output, membrane_output)

	assert affective_output is not None
	assert "emotion_states" in affective_output
	print("✓ Integration test passed")


if __name__ == "__main__":
	print("Running DT-(ESN)-RP Framework Tests\n" + "=" * 50)

	try:
		test_esn_reservoir()
		test_p_system()
		test_affective_agency()
		test_ridge_regression()
		test_julia_ricci_flow()
		test_integration()

		print("\n" + "=" * 50)
		print("All tests passed! ✓")
		print("=" * 50)

	except Exception as e:
		print(f"\n✗ Test failed with error: {e}")
		import traceback

		traceback.print_exc()
		sys.exit(1)
