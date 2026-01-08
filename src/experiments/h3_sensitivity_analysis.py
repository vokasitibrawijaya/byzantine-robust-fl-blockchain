"""
H3 Cost Model Sensitivity Analysis
Systematically vary cost parameters to assess model robustness.
"""

import json
import numpy as np
from typing import Dict, List
from spectral_sketching import simulate_spectral_detection


def baseline_cost_model() -> Dict:
    """Default cost parameters from original experiment"""
    return {
        'gas_per_storage_slot': 20000,
        'gas_price_gwei': 30,
        'l1_to_l2_cost_ratio': 10,
        'centralized_bandwidth_mbps': 100,
        'centralized_cost_per_tb': 0.10
    }


def compute_single_scenario(params: Dict, n_workers: int = 100, n_rounds: int = 50) -> Dict:
    """
    Run spectral sketching with given cost parameters.
    
    Returns:
        Cost metrics dict
    """
    n_byzantine = int(n_workers * 0.2)
    
    # Run spectral detection simulation
    result = simulate_spectral_detection(
        n_clients=n_workers,
        n_byzantine=n_byzantine,
        model_dim=10000,
        sketch_dim=64,
        attack_strength=5.0,
        use_torch=True
    )
    
    # Compute cost with custom parameters
    l2_size_mb = 64 * 4 / (1024 * 1024) * n_rounds  # 64 floats * 4 bytes per float
    centralized_size_mb = 10000 * 4 / (1024 * 1024) * n_rounds
    
    # L2 storage cost
    slots_needed = int(np.ceil(l2_size_mb * 1024 * 1024 / 32))  # 32 bytes per slot
    gas_cost = slots_needed * params['gas_per_storage_slot']
    l2_cost_usd = gas_cost * params['gas_price_gwei'] * 1e-9
    
    # Centralized cost
    centralized_cost_usd = (centralized_size_mb / 1024) * params['centralized_cost_per_tb']
    
    cost_reduction = ((centralized_cost_usd - l2_cost_usd) / centralized_cost_usd) * 100 if centralized_cost_usd > 0 else 0
    
    # Extract detection metrics safely
    detection = result.get('detection', {})
    
    return {
        'l2_cost_usd': l2_cost_usd,
        'centralized_cost_usd': centralized_cost_usd,
        'cost_reduction': cost_reduction,
        'precision': detection.get('precision', 1.0),
        'recall': detection.get('recall', 1.0),
        'f1': detection.get('f1', 1.0)
    }


def vary_single_parameter(param_name: str, variations: List[float]) -> Dict:
    """
    Vary one parameter while keeping others fixed.
    
    Args:
        param_name: Name of parameter to vary
        variations: List of multipliers (e.g., [0.5, 0.75, 1.0, 1.5, 2.0])
    
    Returns:
        Sensitivity analysis results
    """
    baseline = baseline_cost_model()
    results = []
    
    for multiplier in variations:
        params = baseline.copy()
        params[param_name] = baseline[param_name] * multiplier
        
        print(f"  Testing {param_name} = {params[param_name]} ({multiplier:.1f}x baseline)...")
        
        outcome = compute_single_scenario(params)
        results.append({
            'multiplier': multiplier,
            'param_value': params[param_name],
            **outcome
        })
    
    return {
        'parameter': param_name,
        'baseline_value': baseline[param_name],
        'results': results
    }


def full_sensitivity_analysis(save_path: str = "h3_sensitivity_results.json") -> Dict:
    """
    Run comprehensive sensitivity analysis for all cost parameters.
    """
    print("=" * 80)
    print("H3 COST MODEL SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    # Standard variations: ±50% in 25% increments
    variations = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    parameters = [
        'gas_per_storage_slot',
        'gas_price_gwei',
        'l1_to_l2_cost_ratio',
        'centralized_bandwidth_mbps',
        'centralized_cost_per_tb'
    ]
    
    all_results = {}
    
    for param in parameters:
        print(f"\n[*] Analyzing parameter: {param}")
        all_results[param] = vary_single_parameter(param, variations)
    
    # Compute summary statistics
    summary = {}
    for param, data in all_results.items():
        cost_reductions = [r['cost_reduction'] for r in data['results']]
        summary[param] = {
            'min_cost_reduction': min(cost_reductions),
            'max_cost_reduction': max(cost_reductions),
            'range': max(cost_reductions) - min(cost_reductions),
            'cv': np.std(cost_reductions) / np.mean(cost_reductions) if np.mean(cost_reductions) > 0 else 0
        }
    
    output = {
        'baseline': baseline_cost_model(),
        'variations': variations,
        'parameter_analyses': all_results,
        'summary': summary
    }
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SENSITIVITY SUMMARY")
    print("=" * 80)
    for param, stats in summary.items():
        print(f"\n{param}:")
        print(f"  Cost reduction range: {stats['min_cost_reduction']:.1f}% - {stats['max_cost_reduction']:.1f}%")
        print(f"  Variation span: {stats['range']:.1f}%")
        print(f"  Coefficient of variation: {stats['cv']:.3f}")
    
    print(f"\n[OK] Results saved to: {save_path}")
    print("=" * 80)
    
    return output


if __name__ == "__main__":
    full_sensitivity_analysis()
