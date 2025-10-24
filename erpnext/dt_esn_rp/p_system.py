"""
P-System Membrane Computing Implementation

This module implements Paun P-System membrane computing for parallel
distributed computation and reservoir evolution.
"""

import numpy as np
from typing import Dict, List, Optional, Any


class PSystemMembraneComputing:
	"""
	P-System Membrane Computing for distributed parallel computation.
	
	Implements membrane computing principles with multi-level membrane structures
	for evolutionary computation and information processing.
	"""
	
	def __init__(
		self,
		num_membranes: int = 3,
		membrane_capacity: int = 1000,
		evolution_rules: Optional[List[str]] = None,
		enable_membrane_dissolution: bool = True,
	):
		"""
		Initialize P-System Membrane Computing.
		
		Args:
			num_membranes: Number of membrane layers
			membrane_capacity: Maximum capacity per membrane
			evolution_rules: List of evolution rules
			enable_membrane_dissolution: Enable membrane dissolution
		"""
		self.num_membranes = num_membranes
		self.membrane_capacity = membrane_capacity
		self.enable_membrane_dissolution = enable_membrane_dissolution
		
		# Initialize membrane structure
		self.membranes = []
		for i in range(num_membranes):
			membrane = {
				"id": i,
				"level": i,
				"content": np.zeros(membrane_capacity),
				"rules": evolution_rules or self._default_evolution_rules(),
				"active": True,
			}
			self.membranes.append(membrane)
		
		# Membrane communication channels
		self.communication_channels = self._initialize_channels()
	
	def _default_evolution_rules(self) -> List[str]:
		"""
		Define default evolution rules for P-System.
		
		Returns:
			List of evolution rule identifiers
		"""
		return [
			"replication",
			"transformation",
			"communication",
			"dissolution",
		]
	
	def _initialize_channels(self) -> Dict[str, Any]:
		"""
		Initialize communication channels between membranes.
		
		Returns:
			Dictionary of communication channels
		"""
		channels = {}
		for i in range(self.num_membranes - 1):
			channel_id = f"membrane_{i}_to_{i+1}"
			channels[channel_id] = {
				"source": i,
				"target": i + 1,
				"bandwidth": 1.0,
				"active": True,
			}
		return channels
	
	def evolve(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Evolve membrane system based on input state.
		
		Args:
			input_state: Input state from reservoir
			
		Returns:
			Evolved membrane state
		"""
		# Extract reservoir state
		if "reservoir_state" in input_state:
			reservoir_state = input_state["reservoir_state"]
		else:
			reservoir_state = input_state
		
		# Inject input into first membrane
		self._inject_input(reservoir_state)
		
		# Apply evolution rules
		for step in range(self.num_membranes):
			self._apply_evolution_rules(step)
		
		# Collect outputs from all membranes
		outputs = self._collect_outputs()
		
		return {
			"membrane_states": outputs,
			"system_energy": self._compute_system_energy(),
			"active_membranes": sum(1 for m in self.membranes if m["active"]),
		}
	
	def _inject_input(self, reservoir_state: np.ndarray):
		"""
		Inject reservoir state into first membrane.
		
		Args:
			reservoir_state: State from reservoir
		"""
		membrane = self.membranes[0]
		size = min(len(reservoir_state), self.membrane_capacity)
		membrane["content"][:size] = reservoir_state[:size]
	
	def _apply_evolution_rules(self, membrane_index: int):
		"""
		Apply evolution rules to specified membrane.
		
		Args:
			membrane_index: Index of membrane to evolve
		"""
		if membrane_index >= len(self.membranes):
			return
		
		membrane = self.membranes[membrane_index]
		
		if not membrane["active"]:
			return
		
		# Apply rules
		for rule in membrane["rules"]:
			if rule == "replication":
				self._apply_replication(membrane)
			elif rule == "transformation":
				self._apply_transformation(membrane)
			elif rule == "communication":
				self._apply_communication(membrane_index)
			elif rule == "dissolution":
				if self.enable_membrane_dissolution:
					self._apply_dissolution(membrane)
	
	def _apply_replication(self, membrane: Dict[str, Any]):
		"""
		Apply replication rule to membrane.
		
		Args:
			membrane: Target membrane
		"""
		# Simple replication: copy active patterns
		content = membrane["content"]
		active_indices = np.where(np.abs(content) > 0.5)[0]
		
		if len(active_indices) > 0:
			# Replicate high-activation neurons
			for idx in active_indices[:10]:  # Limit replication
				membrane["content"][idx] *= 1.1
	
	def _apply_transformation(self, membrane: Dict[str, Any]):
		"""
		Apply transformation rule to membrane.
		
		Args:
			membrane: Target membrane
		"""
		# Apply non-linear transformation
		membrane["content"] = np.tanh(membrane["content"])
	
	def _apply_communication(self, membrane_index: int):
		"""
		Apply communication between membranes.
		
		Args:
			membrane_index: Source membrane index
		"""
		if membrane_index >= self.num_membranes - 1:
			return
		
		source_membrane = self.membranes[membrane_index]
		target_membrane = self.membranes[membrane_index + 1]
		
		channel_id = f"membrane_{membrane_index}_to_{membrane_index+1}"
		channel = self.communication_channels[channel_id]
		
		if not channel["active"]:
			return
		
		# Transfer information based on bandwidth
		transfer_amount = int(self.membrane_capacity * channel["bandwidth"] * 0.1)
		
		# Find most active neurons in source
		active_indices = np.argsort(np.abs(source_membrane["content"]))[-transfer_amount:]
		
		# Transfer to target
		for idx in active_indices:
			if idx < len(target_membrane["content"]):
				target_membrane["content"][idx] += source_membrane["content"][idx] * 0.5
	
	def _apply_dissolution(self, membrane: Dict[str, Any]):
		"""
		Apply dissolution rule to membrane.
		
		Args:
			membrane: Target membrane
		"""
		# Check dissolution condition (low energy)
		energy = np.sum(membrane["content"] ** 2)
		if energy < 0.1:
			membrane["active"] = False
	
	def _collect_outputs(self) -> List[np.ndarray]:
		"""
		Collect outputs from all membranes.
		
		Returns:
			List of membrane contents
		"""
		return [m["content"].copy() for m in self.membranes]
	
	def _compute_system_energy(self) -> float:
		"""
		Compute total system energy.
		
		Returns:
			System energy value
		"""
		total_energy = 0.0
		for membrane in self.membranes:
			if membrane["active"]:
				total_energy += np.sum(membrane["content"] ** 2)
		return float(total_energy)
	
	def get_membrane_info(self) -> Dict[str, Any]:
		"""
		Get information about membrane system.
		
		Returns:
			Membrane system configuration
		"""
		return {
			"num_membranes": self.num_membranes,
			"membrane_capacity": self.membrane_capacity,
			"active_membranes": sum(1 for m in self.membranes if m["active"]),
			"total_channels": len(self.communication_channels),
			"system_energy": self._compute_system_energy(),
		}
