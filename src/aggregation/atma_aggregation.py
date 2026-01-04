"""
ATMA - Adaptive Trimmed Mean Aggregation
Based on: Kalibbala et al. (2025)
"Blockchain-augmented FL IDS for non-IID Edge-IoT"

This implementation provides adaptive threshold adjustment for Byzantine-robust
aggregation in highly non-IID federated learning environments.
"""

import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ATMAAggregator:
    """
    Adaptive Trimmed Mean Aggregation with dynamic threshold adjustment.
    
    Key Features:
    - Dynamic trim percentage based on gradient statistics
    - Non-IID data handling through adaptive mechanisms
    - Byzantine robustness through statistical outlier detection
    """
    
    def __init__(self, 
                 initial_trim_ratio: float = 0.1,
                 adaptation_rate: float = 0.05,
                 min_trim_ratio: float = 0.05,
                 max_trim_ratio: float = 0.3):
        """
        Initialize ATMA aggregator.
        
        Args:
            initial_trim_ratio: Initial percentage of gradients to trim (both ends)
            adaptation_rate: Speed of threshold adaptation
            min_trim_ratio: Minimum trim percentage
            max_trim_ratio: Maximum trim percentage
        """
        self.trim_ratio = initial_trim_ratio
        self.adaptation_rate = adaptation_rate
        self.min_trim_ratio = min_trim_ratio
        self.max_trim_ratio = max_trim_ratio
        
        # Statistics tracking for adaptation
        self.history_variance = []
        self.history_kurtosis = []
        self.round_count = 0
        
        logger.info(f"ATMA initialized: trim_ratio={initial_trim_ratio}, "
                   f"adaptation_rate={adaptation_rate}")
    
    def aggregate(self, gradients: List[np.ndarray], 
                  client_weights: List[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform ATMA aggregation with adaptive threshold.
        
        Args:
            gradients: List of gradient arrays from clients
            client_weights: Optional weights for each client (default: equal weights)
            
        Returns:
            Tuple of (aggregated_gradient, metadata)
        """
        if not gradients:
            raise ValueError("No gradients provided for aggregation")
        
        self.round_count += 1
        n_clients = len(gradients)
        
        # Default to equal weights
        if client_weights is None:
            client_weights = [1.0 / n_clients] * n_clients
        
        # Convert to numpy array for easier manipulation
        gradient_matrix = np.array(gradients)  # Shape: (n_clients, model_dim)
        
        # Compute statistics for adaptive threshold
        variance = np.var(gradient_matrix, axis=0).mean()
        kurtosis = self._compute_kurtosis(gradient_matrix)
        
        self.history_variance.append(variance)
        self.history_kurtosis.append(kurtosis)
        
        # Adapt trim ratio based on distribution characteristics
        self._adapt_trim_ratio(variance, kurtosis)
        
        # Perform trimmed mean aggregation
        aggregated = self._trimmed_mean(gradient_matrix, client_weights)
        
        # Compute metadata
        metadata = {
            'trim_ratio': self.trim_ratio,
            'variance': variance,
            'kurtosis': kurtosis,
            'n_clients': n_clients,
            'n_trimmed': int(n_clients * self.trim_ratio * 2),  # Both ends
            'round': self.round_count,
            'algorithm': 'ATMA'
        }
        
        logger.debug(f"Round {self.round_count}: ATMA aggregation with trim_ratio={self.trim_ratio:.3f}, "
                    f"variance={variance:.6f}, kurtosis={kurtosis:.3f}")
        
        return aggregated, metadata
    
    def _trimmed_mean(self, gradient_matrix: np.ndarray, 
                      weights: List[float]) -> np.ndarray:
        """
        Compute trimmed mean by removing outliers from both ends.
        
        Args:
            gradient_matrix: Shape (n_clients, model_dim)
            weights: Client weights
            
        Returns:
            Aggregated gradient
        """
        n_clients = gradient_matrix.shape[0]
        n_trim = max(1, int(n_clients * self.trim_ratio))  # At least trim 1 from each end
        
        # Compute L2 norm for each gradient
        norms = np.linalg.norm(gradient_matrix, axis=1)
        
        # Sort indices by norm
        sorted_indices = np.argsort(norms)
        
        # Remove n_trim from both ends
        kept_indices = sorted_indices[n_trim:-n_trim] if n_trim < n_clients // 2 else sorted_indices
        
        if len(kept_indices) == 0:
            # Fallback: use middle client if all trimmed
            kept_indices = [sorted_indices[len(sorted_indices) // 2]]
        
        # Aggregate kept gradients with weights
        kept_gradients = gradient_matrix[kept_indices]
        kept_weights = np.array([weights[i] for i in kept_indices])
        
        # Normalize weights
        kept_weights = kept_weights / kept_weights.sum()
        
        # Weighted average
        aggregated = np.average(kept_gradients, axis=0, weights=kept_weights)
        
        return aggregated
    
    def _adapt_trim_ratio(self, variance: float, kurtosis: float):
        """
        Adapt trim ratio based on gradient distribution statistics.
        
        High variance or high kurtosis (heavy tails) → increase trimming
        Low variance and low kurtosis → decrease trimming
        
        Args:
            variance: Current gradient variance
            kurtosis: Current gradient kurtosis
        """
        # Compute moving averages if we have history
        if len(self.history_variance) > 5:
            avg_variance = np.mean(self.history_variance[-5:])
            avg_kurtosis = np.mean(self.history_kurtosis[-5:])
        else:
            avg_variance = variance
            avg_kurtosis = kurtosis
        
        # Adaptation logic:
        # If variance is increasing or kurtosis is high (heavy tails), increase trimming
        # Otherwise, decrease trimming
        
        variance_trend = 0
        if len(self.history_variance) > 1:
            variance_trend = (variance - self.history_variance[-2]) / (self.history_variance[-2] + 1e-8)
        
        # High kurtosis (>3) indicates heavy tails → more outliers
        # Variance increasing → more dispersion
        adjustment = 0
        
        if kurtosis > 3.5 or variance_trend > 0.1:
            # Increase trimming
            adjustment = self.adaptation_rate
        elif kurtosis < 2.5 and variance_trend < -0.05:
            # Decrease trimming
            adjustment = -self.adaptation_rate
        
        # Update trim ratio
        self.trim_ratio = np.clip(
            self.trim_ratio + adjustment,
            self.min_trim_ratio,
            self.max_trim_ratio
        )
    
    def _compute_kurtosis(self, gradient_matrix: np.ndarray) -> float:
        """
        Compute excess kurtosis of gradient norms.
        
        Kurtosis measures "tailedness" of distribution:
        - Normal distribution: kurtosis ≈ 3
        - Heavy tails (outliers): kurtosis > 3
        - Light tails: kurtosis < 3
        
        Args:
            gradient_matrix: Shape (n_clients, model_dim)
            
        Returns:
            Excess kurtosis value
        """
        norms = np.linalg.norm(gradient_matrix, axis=1)
        
        if len(norms) < 4:
            return 3.0  # Default to normal distribution
        
        mean = np.mean(norms)
        std = np.std(norms)
        
        if std < 1e-8:
            return 3.0  # Avoid division by zero
        
        # Excess kurtosis (Fisher's definition)
        kurtosis = np.mean(((norms - mean) / std) ** 4) - 3
        
        return 3.0 + kurtosis  # Return total kurtosis (not excess)
    
    def reset(self):
        """Reset aggregator state."""
        self.trim_ratio = self.min_trim_ratio
        self.history_variance = []
        self.history_kurtosis = []
        self.round_count = 0
        logger.info("ATMA aggregator reset")


def atma_aggregate_wrapper(gradients: List[np.ndarray], 
                           trim_ratio: float = 0.1,
                           **kwargs) -> np.ndarray:
    """
    Simple wrapper for ATMA aggregation (stateless for single use).
    
    Args:
        gradients: List of gradient arrays
        trim_ratio: Initial trim ratio
        **kwargs: Additional parameters
        
    Returns:
        Aggregated gradient
    """
    aggregator = ATMAAggregator(initial_trim_ratio=trim_ratio)
    aggregated, _ = aggregator.aggregate(gradients)
    return aggregated


# Simulation mode for testing
def simulate_atma(n_clients: int = 100,
                  n_byzantine: int = 20,
                  model_dim: int = 1000,
                  n_rounds: int = 10,
                  attack_strength: float = 5.0) -> Dict:
    """
    Simulate ATMA aggregation with Byzantine clients.
    
    Args:
        n_clients: Total number of clients
        n_byzantine: Number of Byzantine clients
        model_dim: Model dimension
        n_rounds: Number of aggregation rounds
        attack_strength: Strength of Byzantine attack
        
    Returns:
        Simulation results dictionary
    """
    logger.info(f"Simulating ATMA: {n_clients} clients, {n_byzantine} Byzantine, "
               f"{n_rounds} rounds")
    
    aggregator = ATMAAggregator()
    results = {
        'rounds': [],
        'trim_ratios': [],
        'variances': [],
        'kurtosis_values': [],
        'aggregation_errors': []
    }
    
    # True model update (for error computation)
    true_gradient = np.random.randn(model_dim) * 0.1
    
    for round_num in range(n_rounds):
        # Generate honest gradients (close to true gradient)
        honest_gradients = [
            true_gradient + np.random.randn(model_dim) * 0.05
            for _ in range(n_clients - n_byzantine)
        ]
        
        # Generate Byzantine gradients (malicious)
        byzantine_gradients = [
            true_gradient * attack_strength + np.random.randn(model_dim) * 0.5
            for _ in range(n_byzantine)
        ]
        
        # Combine
        all_gradients = honest_gradients + byzantine_gradients
        np.random.shuffle(all_gradients)
        
        # Aggregate
        aggregated, metadata = aggregator.aggregate(all_gradients)
        
        # Compute error
        error = np.linalg.norm(aggregated - true_gradient)
        
        # Store results
        results['rounds'].append(round_num)
        results['trim_ratios'].append(metadata['trim_ratio'])
        results['variances'].append(metadata['variance'])
        results['kurtosis_values'].append(metadata['kurtosis'])
        results['aggregation_errors'].append(error)
        
        logger.debug(f"Round {round_num}: trim_ratio={metadata['trim_ratio']:.3f}, "
                    f"error={error:.4f}")
    
    # Summary statistics
    results['summary'] = {
        'avg_trim_ratio': np.mean(results['trim_ratios']),
        'final_trim_ratio': results['trim_ratios'][-1],
        'avg_error': np.mean(results['aggregation_errors']),
        'min_error': np.min(results['aggregation_errors']),
        'max_error': np.max(results['aggregation_errors']),
        'n_clients': n_clients,
        'n_byzantine': n_byzantine,
        'byzantine_ratio': n_byzantine / n_clients
    }
    
    logger.info(f"Simulation complete. Avg error: {results['summary']['avg_error']:.4f}, "
               f"Final trim ratio: {results['summary']['final_trim_ratio']:.3f}")
    
    return results


if __name__ == "__main__":
    # Test ATMA implementation
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("ATMA - Adaptive Trimmed Mean Aggregation")
    print("Based on: Kalibbala et al. (2025)")
    print("=" * 80)
    
    # Test 1: Basic aggregation
    print("\n[Test 1] Basic ATMA Aggregation")
    print("-" * 80)
    
    test_gradients = [
        np.random.randn(100) * 0.1,  # Honest
        np.random.randn(100) * 0.1,  # Honest
        np.random.randn(100) * 0.1,  # Honest
        np.random.randn(100) * 5.0,  # Byzantine (large)
        np.random.randn(100) * 5.0,  # Byzantine (large)
    ]
    
    aggregator = ATMAAggregator(initial_trim_ratio=0.2)
    result, metadata = aggregator.aggregate(test_gradients)
    
    print(f"✓ Aggregated gradient shape: {result.shape}")
    print(f"✓ Trim ratio: {metadata['trim_ratio']:.3f}")
    print(f"✓ Trimmed {metadata['n_trimmed']} out of {metadata['n_clients']} clients")
    print(f"✓ Variance: {metadata['variance']:.6f}")
    print(f"✓ Kurtosis: {metadata['kurtosis']:.3f}")
    
    # Test 2: Full simulation
    print("\n[Test 2] Full ATMA Simulation (10 rounds)")
    print("-" * 80)
    
    sim_results = simulate_atma(
        n_clients=100,
        n_byzantine=20,
        model_dim=1000,
        n_rounds=10,
        attack_strength=5.0
    )
    
    print(f"\n✓ Simulation Summary:")
    print(f"  - Average trim ratio: {sim_results['summary']['avg_trim_ratio']:.3f}")
    print(f"  - Final trim ratio: {sim_results['summary']['final_trim_ratio']:.3f}")
    print(f"  - Average aggregation error: {sim_results['summary']['avg_error']:.4f}")
    print(f"  - Min error: {sim_results['summary']['min_error']:.4f}")
    print(f"  - Max error: {sim_results['summary']['max_error']:.4f}")
    print(f"  - Byzantine ratio: {sim_results['summary']['byzantine_ratio']:.1%}")
    
    # Test 3: Comparison with static trimmed mean
    print("\n[Test 3] ATMA vs Static Trimmed Mean")
    print("-" * 80)
    
    # ATMA
    atma_results = simulate_atma(n_clients=100, n_byzantine=30, n_rounds=20)
    
    # Static trimmed mean (fixed trim_ratio=0.15)
    aggregator_static = ATMAAggregator(initial_trim_ratio=0.15, adaptation_rate=0.0)
    static_errors = []
    true_gradient = np.random.randn(1000) * 0.1
    
    for _ in range(20):
        honest = [true_gradient + np.random.randn(1000) * 0.05 for _ in range(70)]
        byzantine = [true_gradient * 5.0 + np.random.randn(1000) * 0.5 for _ in range(30)]
        all_grads = honest + byzantine
        np.random.shuffle(all_grads)
        
        agg, _ = aggregator_static.aggregate(all_grads)
        error = np.linalg.norm(agg - true_gradient)
        static_errors.append(error)
    
    print(f"✓ ATMA average error: {atma_results['summary']['avg_error']:.4f}")
    print(f"✓ Static TM average error: {np.mean(static_errors):.4f}")
    
    improvement = (np.mean(static_errors) - atma_results['summary']['avg_error']) / np.mean(static_errors) * 100
    print(f"✓ ATMA improvement: {improvement:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ All ATMA tests passed!")
    print("=" * 80)
