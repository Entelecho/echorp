"""
Affective Agency and Emotional Intelligence Framework

This module implements the Differential Emotion Theory Framework with
Affective Agency for emotional resonance and cognitive attention mechanisms.
"""

import numpy as np
from typing import Dict, List, Optional, Any


class AffectiveAgency:
	"""
	Affective Agency for emotional intelligence and resonance.
	
	Implements Differential Emotion Theory Framework integrated with
	reservoir dynamics and cognitive attention mechanisms.
	"""
	
	# Basic emotions based on Differential Emotion Theory
	BASIC_EMOTIONS = [
		"interest",
		"joy",
		"surprise",
		"sadness",
		"anger",
		"disgust",
		"contempt",
		"fear",
		"shame",
		"guilt",
	]
	
	def __init__(
		self,
		enable_cognitive_attention: bool = True,
		attention_mechanism: str = "transformer",
		emotion_decay_rate: float = 0.1,
	):
		"""
		Initialize Affective Agency.
		
		Args:
			enable_cognitive_attention: Enable cognitive attention mechanism
			attention_mechanism: Type of attention mechanism ("transformer", "simple")
			emotion_decay_rate: Rate of emotion decay over time
		"""
		self.enable_cognitive_attention = enable_cognitive_attention
		self.attention_mechanism = attention_mechanism
		self.emotion_decay_rate = emotion_decay_rate
		
		# Initialize emotion states
		self.emotion_states = {
			emotion: 0.0 for emotion in self.BASIC_EMOTIONS
		}
		
		# Emotion valence and arousal mapping
		self.emotion_valence = {
			"interest": 0.5,
			"joy": 1.0,
			"surprise": 0.3,
			"sadness": -0.8,
			"anger": -0.7,
			"disgust": -0.6,
			"contempt": -0.5,
			"fear": -0.9,
			"shame": -0.7,
			"guilt": -0.6,
		}
		
		self.emotion_arousal = {
			"interest": 0.6,
			"joy": 0.8,
			"surprise": 0.9,
			"sadness": 0.3,
			"anger": 0.9,
			"disgust": 0.5,
			"contempt": 0.4,
			"fear": 0.9,
			"shame": 0.6,
			"guilt": 0.5,
		}
	
	def process_emotions(
		self,
		reservoir_state: Dict[str, Any],
		membrane_state: Dict[str, Any],
	) -> Dict[str, Any]:
		"""
		Process emotional states based on reservoir and membrane dynamics.
		
		Args:
			reservoir_state: State from ESN reservoir
			membrane_state: State from P-System membranes
			
		Returns:
			Processed emotional state and affective resonance
		"""
		# Extract activation patterns
		activation_energy = reservoir_state.get("activation_energy", 0.0)
		system_energy = membrane_state.get("system_energy", 0.0)
		
		# Update emotion states based on energy dynamics
		self._update_emotions(activation_energy, system_energy)
		
		# Compute affective resonance
		resonance = self._compute_affective_resonance()
		
		# Apply cognitive attention if enabled
		attention_weights = None
		if self.enable_cognitive_attention:
			attention_weights = self._compute_attention_weights(reservoir_state)
		
		return {
			"emotion_states": self.emotion_states.copy(),
			"affective_resonance": resonance,
			"valence": self._compute_overall_valence(),
			"arousal": self._compute_overall_arousal(),
			"attention_weights": attention_weights,
		}
	
	def _update_emotions(self, activation_energy: float, system_energy: float):
		"""
		Update emotion states based on system dynamics.
		
		Args:
			activation_energy: Energy from reservoir activation
			system_energy: Energy from membrane system
		"""
		# Normalize energies
		total_energy = activation_energy + system_energy
		normalized_energy = np.tanh(total_energy / 100.0)
		
		# Update emotions based on energy patterns
		# High energy -> arousal emotions (joy, anger, fear, surprise)
		# Low energy -> low arousal emotions (sadness, contentment)
		
		if normalized_energy > 0.5:
			self.emotion_states["joy"] += 0.1 * (normalized_energy - 0.5)
			self.emotion_states["interest"] += 0.05 * (normalized_energy - 0.5)
			self.emotion_states["surprise"] += 0.08 * (normalized_energy - 0.5)
		elif normalized_energy < -0.5:
			self.emotion_states["sadness"] += 0.1 * abs(normalized_energy + 0.5)
			self.emotion_states["fear"] += 0.05 * abs(normalized_energy + 0.5)
		
		# Apply decay to all emotions
		for emotion in self.emotion_states:
			self.emotion_states[emotion] *= (1 - self.emotion_decay_rate)
			# Clamp between 0 and 1
			self.emotion_states[emotion] = np.clip(self.emotion_states[emotion], 0.0, 1.0)
	
	def _compute_affective_resonance(self) -> float:
		"""
		Compute overall affective resonance.
		
		Returns:
			Affective resonance value
		"""
		# Resonance is based on emotion coherence and intensity
		total_intensity = sum(self.emotion_states.values())
		
		if total_intensity == 0:
			return 0.0
		
		# Compute emotion variance (coherence measure)
		emotion_values = list(self.emotion_states.values())
		variance = np.var(emotion_values)
		
		# High resonance = high intensity, low variance (coherent emotions)
		resonance = total_intensity * (1.0 / (1.0 + variance))
		
		return float(resonance)
	
	def _compute_overall_valence(self) -> float:
		"""
		Compute overall emotional valence.
		
		Returns:
			Valence value (-1 to 1, negative to positive)
		"""
		weighted_valence = 0.0
		total_weight = 0.0
		
		for emotion, intensity in self.emotion_states.items():
			weighted_valence += self.emotion_valence[emotion] * intensity
			total_weight += intensity
		
		if total_weight == 0:
			return 0.0
		
		return weighted_valence / total_weight
	
	def _compute_overall_arousal(self) -> float:
		"""
		Compute overall emotional arousal.
		
		Returns:
			Arousal value (0 to 1, calm to excited)
		"""
		weighted_arousal = 0.0
		total_weight = 0.0
		
		for emotion, intensity in self.emotion_states.items():
			weighted_arousal += self.emotion_arousal[emotion] * intensity
			total_weight += intensity
		
		if total_weight == 0:
			return 0.0
		
		return weighted_arousal / total_weight
	
	def _compute_attention_weights(
		self,
		reservoir_state: Dict[str, Any],
	) -> np.ndarray:
		"""
		Compute cognitive attention weights.
		
		Args:
			reservoir_state: State from ESN reservoir
			
		Returns:
			Attention weight vector
		"""
		state = reservoir_state.get("reservoir_state", np.array([]))
		
		if len(state) == 0:
			return np.array([])
		
		if self.attention_mechanism == "transformer":
			# Simplified transformer-style attention
			# Q = K = V = reservoir_state
			attention_scores = np.dot(state, state) / np.sqrt(len(state))
			attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
		else:
			# Simple attention based on activation magnitude
			attention_weights = np.abs(state) / (np.sum(np.abs(state)) + 1e-8)
		
		return attention_weights
	
	def get_emotion_summary(self) -> Dict[str, Any]:
		"""
		Get summary of emotional state.
		
		Returns:
			Emotional state summary
		"""
		# Find dominant emotion
		dominant_emotion = max(self.emotion_states.items(), key=lambda x: x[1])
		
		return {
			"dominant_emotion": dominant_emotion[0],
			"dominant_intensity": dominant_emotion[1],
			"valence": self._compute_overall_valence(),
			"arousal": self._compute_overall_arousal(),
			"resonance": self._compute_affective_resonance(),
			"all_emotions": self.emotion_states.copy(),
		}
