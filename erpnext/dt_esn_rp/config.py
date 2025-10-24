"""
Configuration and Integration for DT-(ESN)-RP Framework

This module provides configuration management and integration hooks
for the Deep Tree Echo State Network Reservoir P-System framework.
"""

import frappe
from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
	"reservoir": {
		"size": 1000,
		"spectral_radius": 0.9,
		"input_scaling": 1.0,
		"leak_rate": 0.3,
		"connectivity": 0.1,
	},
	"p_system": {
		"num_membranes": 3,
		"membrane_capacity": 1000,
		"enable_dissolution": True,
	},
	"affective_agency": {
		"enable": True,
		"attention_mechanism": "transformer",
		"emotion_decay_rate": 0.1,
	},
	"ricci_flow": {
		"enable": True,
		"dimension": 3,
		"flow_rate": 0.01,
	},
	"ridge_regression": {
		"regularization": 1e-6,
		"integration_method": "rk4",
		"use_gradient_descent": False,
	},
}


def get_framework_config() -> Dict[str, Any]:
	"""
	Get DT-(ESN)-RP framework configuration.
	
	Returns configuration from site config or defaults.
	
	Returns:
		Configuration dictionary
	"""
	site_config = frappe.get_site_config()
	dt_esn_rp_config = site_config.get("dt_esn_rp", {})
	
	# Merge with defaults
	config = DEFAULT_CONFIG.copy()
	for key in config:
		if key in dt_esn_rp_config:
			config[key].update(dt_esn_rp_config[key])
	
	return config


def initialize_framework(config: Optional[Dict[str, Any]] = None) -> "DeepTreeESNRP":
	"""
	Initialize DT-(ESN)-RP framework with configuration.
	
	Args:
		config: Optional configuration dictionary
		
	Returns:
		Initialized framework instance
	"""
	from .core import DeepTreeESNRP
	
	if config is None:
		config = get_framework_config()
	
	reservoir_config = config["reservoir"]
	p_system_config = config["p_system"]
	affective_config = config["affective_agency"]
	
	framework = DeepTreeESNRP(
		reservoir_size=reservoir_config["size"],
		spectral_radius=reservoir_config["spectral_radius"],
		input_scaling=reservoir_config["input_scaling"],
		leak_rate=reservoir_config["leak_rate"],
		membrane_layers=p_system_config["num_membranes"],
		enable_affective_resonance=affective_config["enable"],
	)
	
	frappe.logger().info("DT-(ESN)-RP Framework initialized with configuration")
	
	return framework


def get_framework_status() -> Dict[str, Any]:
	"""
	Get status information about the framework.
	
	Returns:
		Status dictionary
	"""
	config = get_framework_config()
	
	return {
		"enabled": True,
		"version": "1.0.0",
		"configuration": config,
		"components": {
			"echo_state_network": True,
			"p_system_membrane": True,
			"affective_agency": config["affective_agency"]["enable"],
			"ricci_flow": config["ricci_flow"]["enable"],
			"ridge_regression": True,
		},
	}


@frappe.whitelist()
def get_framework_info():
	"""
	API endpoint to get framework information.
	
	Returns:
		Framework information as JSON
	"""
	return get_framework_status()


@frappe.whitelist()
def process_with_framework(input_data: str):
	"""
	API endpoint to process data through the framework.
	
	Args:
		input_data: JSON string of input data
		
	Returns:
		Processing results as JSON
	"""
	import json
	import numpy as np
	
	# Parse input
	data = json.loads(input_data)
	input_array = np.array(data)
	
	# Initialize framework
	framework = initialize_framework()
	
	# Process data
	result = framework.process(input_array)
	
	# Convert numpy arrays to lists for JSON serialization
	serializable_result = {}
	for key, value in result.items():
		if isinstance(value, np.ndarray):
			serializable_result[key] = value.tolist()
		elif isinstance(value, dict):
			serializable_result[key] = {}
			for k, v in value.items():
				if isinstance(v, np.ndarray):
					serializable_result[key][k] = v.tolist()
				else:
					serializable_result[key][k] = v
		else:
			serializable_result[key] = value
	
	return serializable_result
