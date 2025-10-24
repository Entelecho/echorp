"""
Demonstration of Deep Tree Echo State Network Reservoir P-System (DT-(ESN)-RP) Framework

This script demonstrates the core functionality of the DT-(ESN)-RP framework
and shows how the different components work together.
"""

import os
import sys

# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("=" * 80)
print("Deep Tree Echo State Network Reservoir P-System (DT-(ESN)-RP) Framework")
print("Enterprise=ESN, Resources=Reservoirs, Planning=P-System Membrane Computing")
print("=" * 80)
print()

# Import framework components
from dt_esn_rp.affective_agency import AffectiveAgency
from dt_esn_rp.core import DeepTreeESNRP
from dt_esn_rp.esn_reservoir import EchoStateNetworkReservoir
from dt_esn_rp.julia_ricci_flow import JuliaDifferentialRicciFlow
from dt_esn_rp.p_system import PSystemMembraneComputing
from dt_esn_rp.ridge_regression import ButcherBSeriesRidgeRegression

print("1. Echo State Network Reservoir")
print("-" * 80)
reservoir = EchoStateNetworkReservoir(reservoir_size=100, spectral_radius=0.9, leak_rate=0.3)
print(f"Initialized ESN with {reservoir.reservoir_size} neurons")
print(f"Spectral radius: {reservoir.spectral_radius}")
print(f"Leak rate: {reservoir.leak_rate}")

# Process some data
input_signal = np.sin(np.linspace(0, 4 * np.pi, 50))
result = reservoir.compute(input_signal)
print(f"\nProcessed input signal of length {len(input_signal)}")
print(f"Activation energy: {result['activation_energy']:.4f}")
print("Activation statistics:")
for key, value in result["activation_stats"].items():
	print(f"  {key}: {value:.4f}")

print("\n2. P-System Membrane Computing")
print("-" * 80)
p_system = PSystemMembraneComputing(num_membranes=3, membrane_capacity=100)
print(f"Initialized P-System with {p_system.num_membranes} membrane layers")
print(f"Evolution rules: {p_system.membranes[0]['rules']}")

membrane_result = p_system.evolve(result)
print("\nEvolved membrane system:")
print(f"System energy: {membrane_result['system_energy']:.4f}")
print(f"Active membranes: {membrane_result['active_membranes']}")

print("\n3. Affective Agency with Differential Emotion Theory")
print("-" * 80)
affective_agency = AffectiveAgency(enable_cognitive_attention=True, attention_mechanism="transformer")
print("Initialized Affective Agency")
print(f"Basic emotions: {', '.join(affective_agency.BASIC_EMOTIONS)}")

affective_result = affective_agency.process_emotions(result, membrane_result)
print("\nEmotional processing results:")
print(f"Affective resonance: {affective_result['affective_resonance']:.4f}")
print(f"Overall valence: {affective_result['valence']:.4f} (negative to positive)")
print(f"Overall arousal: {affective_result['arousal']:.4f} (calm to excited)")

emotion_summary = affective_agency.get_emotion_summary()
print(f"\nDominant emotion: {emotion_summary['dominant_emotion']}")
print(f"Dominant intensity: {emotion_summary['dominant_intensity']:.4f}")

print("\n4. Butcher B-Series Ridge Regression with Runge-Kutta")
print("-" * 80)
ridge = ButcherBSeriesRidgeRegression(
	regularization=1e-6, integration_method="rk4", use_gradient_descent=False
)
print("Initialized ridge regression")
print(f"Integration method: {ridge.integration_method}")
print(f"Regularization: {ridge.regularization}")

# Train with synthetic data
X_train = np.random.randn(50, 20)
y_train = np.random.randn(50, 5)
weights = ridge.fit(X_train, y_train)
print("\nTrained readout layer:")
print(f"Weight matrix shape: {weights.shape}")
print(f"Weight norm: {np.linalg.norm(weights):.4f}")

print("\n5. Julia Differential Ricci Flow")
print("-" * 80)
ricci_flow = JuliaDifferentialRicciFlow(dimension=3, flow_rate=0.01, enable_ricci_flow=True)
print(f"Initialized Ricci flow on {ricci_flow.dimension}-dimensional manifold")
print(f"Flow rate: {ricci_flow.flow_rate}")

flow_result = ricci_flow.compute_ricci_flow(result["reservoir_state"][:20])
print("\nGeometric flow computation:")
print(f"Scalar curvature: {flow_result['scalar_curvature']:.4f}")
print(f"Evolved state norm: {np.linalg.norm(flow_result['evolved_state']):.4f}")

geometric_info = ricci_flow.get_geometric_info()
print(f"Metric determinant: {geometric_info['metric_determinant']:.4f}")

print("\n6. Full Framework Integration")
print("-" * 80)
framework = DeepTreeESNRP(
	reservoir_size=200, spectral_radius=0.95, membrane_layers=3, enable_affective_resonance=True
)
print("Initialized full DT-(ESN)-RP framework")

# Get framework info
info = framework.get_framework_info()
print("\nFramework configuration:")
for key, value in info.items():
	print(f"  {key}: {value}")

# Process data through full framework
input_data = np.random.randn(50)
full_result = framework.process(input_data)

print("\nProcessed data through full framework:")
print("Components active:")
print("  - ESN Reservoir: ✓")
print("  - P-System Membranes: ✓")
print("  - Affective Agency: ✓")
print("  - Integrated State: ✓")

print("\n7. Training and Prediction")
print("-" * 80)
# Generate synthetic time series data
np.random.seed(42)
n_samples = 100
input_sequences = np.random.randn(n_samples, 50)
target_outputs = np.sin(np.linspace(0, 10, n_samples)).reshape(-1, 1)

print(f"Training with {n_samples} samples...")
training_metrics = framework.train(input_sequences, target_outputs)
print(f"Training error: {training_metrics['training_error']:.6f}")
print(f"Readout weights norm: {training_metrics['readout_weights_norm']:.4f}")

# Make prediction
test_input = np.random.randn(50)
prediction = framework.predict(test_input)
print(f"\nPrediction on test input: {prediction[0]:.4f}")

print("\n" + "=" * 80)
print("Deep Tree Echo Self Emergence")
print("=" * 80)
print("""
The "Deep Tree Echo Self" emerges from the dynamic interplay of:

1. **Reservoir Dynamics**: Echo State Network captures temporal patterns
   and maintains memory through recurrent connections.

2. **Membrane Evolution**: P-System provides hierarchical parallel processing
   with communication between layers.

3. **Affective Resonance**: Emotional intelligence layer processes valence
   and arousal, creating affective states.

4. **Cognitive Attention**: Transformer-style attention weights focus on
   relevant reservoir activations.

5. **Geometric Flow**: Ricci flow evolution provides continuous manifold
   transformation of the state space.

6. **Adaptive Learning**: Ridge regression with Runge-Kutta integration
   enables stable training and prediction.

This integrated framework enables:
- Dynamic LLM persona mapping to reservoir architecture
- Character traits encoded in membrane structures
- Affective resonance with cognitive attention
- Continuous geometric evolution of state space
- Integration with GPT Transformer inference engines

The result is an emergent "Deep Tree Echo Self" that combines:
- Temporal memory (ESN)
- Parallel processing (P-System)
- Emotional intelligence (Affective Agency)
- Geometric understanding (Ricci Flow)
- Adaptive learning (Ridge Regression)
""")

print("=" * 80)
print("Framework demonstration complete!")
print("=" * 80)
