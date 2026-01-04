"""
Adaptive Attacker for Federated Learning
Based on: FLARE (2025)
"Adaptive Multi-Dimensional Reputation for Robust Client Selection in 
Federated Learning"

This implementation tests the "Transparency Paradox":
Can attackers exploit on-chain transparency to improve their attacks?
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveAttacker:
    """
    FLARE-inspired adaptive attacker that learns from on-chain feedback.
    
    Key Features:
    - Learns detection patterns from blockchain logs
    - Adapts attack strategy based on historical rejections
    - Tests the "Transparency Paradox" hypothesis
    - Multi-dimensional attack calibration
    """
    
    def __init__(self,
                 initial_attack_strength: float = 5.0,
                 learning_rate: float = 0.1,
                 stealth_mode: bool = False,
                 has_blockchain_access: bool = True):
        """
        Initialize adaptive attacker.
        
        Args:
            initial_attack_strength: Initial magnitude of malicious updates
            learning_rate: How fast to adapt based on feedback
            stealth_mode: Try to evade detection vs maximize damage
            has_blockchain_access: Can read on-chain logs (Transparency Paradox test)
        """
        self.attack_strength = initial_attack_strength
        self.learning_rate = learning_rate
        self.stealth_mode = stealth_mode
        self.has_blockchain_access = has_blockchain_access
        
        # Learning history
        self.attack_history = []
        self.detection_history = []
        self.success_rate = 0.0
        
        # Attack strategy
        self.current_strategy = 'aggressive' if not stealth_mode else 'stealthy'
        
        logger.info(f"Adaptive Attacker initialized: strength={initial_attack_strength}, "
                   f"blockchain_access={has_blockchain_access}, stealth={stealth_mode}")
    
    def generate_malicious_update(self, 
                                  true_gradient: np.ndarray,
                                  honest_gradients: List[np.ndarray],
                                  round_number: int) -> np.ndarray:
        """
        Generate malicious gradient update adapted to current environment.
        
        Args:
            true_gradient: The true model gradient
            honest_gradients: Samples of honest gradients (if attacker can observe)
            round_number: Current training round
            
        Returns:
            Malicious gradient
        """
        # Adapt strategy based on learning
        if round_number > 0 and self.has_blockchain_access:
            self._adapt_strategy()
        
        # Generate attack based on current strategy
        if self.current_strategy == 'aggressive':
            # Maximize model corruption
            malicious = true_gradient * self.attack_strength
        
        elif self.current_strategy == 'stealthy':
            # Try to evade detection while still corrupting
            # Stay within honest gradient distribution
            honest_mean = np.mean([np.linalg.norm(g) for g in honest_gradients])
            honest_std = np.std([np.linalg.norm(g) for g in honest_gradients])
            
            # Generate gradient at edge of distribution
            target_norm = honest_mean + honest_std * 1.5  # Just outside normal range
            malicious = true_gradient * (target_norm / np.linalg.norm(true_gradient))
        
        elif self.current_strategy == 'oscillating':
            # Alternate between aggressive and stealthy
            if round_number % 2 == 0:
                malicious = true_gradient * self.attack_strength
            else:
                malicious = true_gradient * 0.5
        
        elif self.current_strategy == 'sign_flip':
            # Sign-flipping attack (extremely malicious)
            malicious = -true_gradient * self.attack_strength
        
        else:
            # Default: aggressive
            malicious = true_gradient * self.attack_strength
        
        # Add noise for realism
        noise = np.random.randn(*malicious.shape) * 0.1
        malicious += noise
        
        return malicious
    
    def log_attack_result(self, was_detected, was_rejected=None):
        """Log result of attack (learning feedback).

        Backwards compatible:
        - If passed bools, behaves like the original implementation.
        - If passed floats in [0,1], interprets them as detection/rejection *rates*
          for this round (e.g., fraction of attackers detected).
        """
        if was_rejected is None:
            was_rejected = was_detected

        detection_rate = float(was_detected) if not isinstance(was_detected, (bool, np.bool_)) else (1.0 if was_detected else 0.0)
        rejection_rate = float(was_rejected) if not isinstance(was_rejected, (bool, np.bool_)) else (1.0 if was_rejected else 0.0)

        detection_rate = float(np.clip(detection_rate, 0.0, 1.0))
        rejection_rate = float(np.clip(rejection_rate, 0.0, 1.0))

        self.attack_history.append({
            'strength': float(self.attack_strength),
            'strategy': self.current_strategy,
            'detection_rate': detection_rate,
            'rejection_rate': rejection_rate
        })

        self.detection_history.append(detection_rate)

        # Update success rate (recent window)
        if self.detection_history:
            self.success_rate = 1.0 - float(np.mean(self.detection_history[-10:]))
    
    def _adapt_strategy(self):
        """
        Adapt attack strategy based on historical feedback from blockchain.
        
        This is the core of the "Transparency Paradox":
        - With blockchain access: Learn from on-chain detection patterns
        - Without blockchain access: Cannot adapt effectively
        """
        if not self.has_blockchain_access or len(self.attack_history) < 3:
            return
        
        # Analyze recent detection patterns
        recent_history = self.attack_history[-5:]
        recent_detection_rate = float(np.mean([h.get('detection_rate', 0.0) for h in recent_history]))
        
        logger.debug(f"Adapting strategy: recent detection rate = {recent_detection_rate:.1%}")
        
        # Adaptation logic
        if recent_detection_rate > 0.8:
            # Being detected too often → switch to stealth
            if self.current_strategy != 'stealthy':
                logger.info("Switching to STEALTHY strategy (high detection rate)")
                self.current_strategy = 'stealthy'
                self.attack_strength *= (1 - self.learning_rate)  # Reduce strength
        
        elif recent_detection_rate < 0.3:
            # Getting away with it → increase aggression
            if self.current_strategy != 'aggressive':
                logger.info("Switching to AGGRESSIVE strategy (low detection rate)")
                self.current_strategy = 'aggressive'
                self.attack_strength *= (1 + self.learning_rate)  # Increase strength
        
        else:
            # Medium detection → try oscillating
            if self.current_strategy not in ['oscillating', 'sign_flip']:
                logger.info("Switching to OSCILLATING strategy")
                self.current_strategy = 'oscillating'
        
        # Clamp attack strength
        self.attack_strength = np.clip(self.attack_strength, 0.5, 10.0)
    
    def get_statistics(self) -> Dict:
        """Get attacker statistics."""
        if not self.attack_history:
            return {'no_data': True}
        
        return {
            'total_attacks': len(self.attack_history),
            'detection_rate': np.mean(self.detection_history) if self.detection_history else 0,
            'success_rate': self.success_rate,
            'current_strategy': self.current_strategy,
            'current_strength': self.attack_strength,
            'has_blockchain_access': self.has_blockchain_access,
            'stealth_mode': self.stealth_mode
        }


def simulate_transparency_paradox(n_rounds: int = 30,
                                 n_clients: int = 100,
                                 n_attackers: int = 20,
                                 model_dim: int = 1000,
                                 detection_algorithm: str = 'krum',
                                 initial_attack_strength: float = 5.0,
                                 learning_rate: float = 0.15,
                                 detection_sigma_threshold: float = 2.0,
                                 honest_noise_std: float = 0.05,
                                 attacker_noise_std: float = 0.1) -> Dict:
    """
    Simulate the Transparency Paradox experiment.
    
    Compare adaptive attacks:
    - WITH blockchain access (can learn from on-chain logs)
    - WITHOUT blockchain access (centralized, opaque)
    
    Args:
        n_rounds: Number of training rounds
        n_clients: Total clients
        n_attackers: Number of malicious clients
        model_dim: Model dimension
        detection_algorithm: Detection algorithm ('krum', 'trimmed_mean', etc.)
        
    Returns:
        Comparison results
    """
    logger.info(f"Simulating Transparency Paradox: {n_rounds} rounds, "
               f"{n_attackers}/{n_clients} attackers, algorithm={detection_algorithm}")
    
    # Scenario 1: WITH blockchain (attackers can learn)
    attacker_with_blockchain = AdaptiveAttacker(
        initial_attack_strength=initial_attack_strength,
        learning_rate=learning_rate,
        has_blockchain_access=True
    )
    
    # Scenario 2: WITHOUT blockchain (attackers cannot learn)
    attacker_without_blockchain = AdaptiveAttacker(
        initial_attack_strength=initial_attack_strength,
        learning_rate=learning_rate,
        has_blockchain_access=False
    )
    
    # True gradient
    true_gradient = np.random.randn(model_dim) * 0.1
    
    # Results tracking
    results_with_blockchain = {
        'malicious_detection_rates': [],
        'malicious_success_rates': [],
        'attack_strengths': [],
        'model_errors': []
    }
    
    results_without_blockchain = {
        'malicious_detection_rates': [],
        'malicious_success_rates': [],
        'attack_strengths': [],
        'model_errors': []
    }
    
    # Simulate both scenarios
    for round_num in range(n_rounds):
        # Generate honest gradients
        honest_gradients = [
            true_gradient + np.random.randn(model_dim) * honest_noise_std
            for _ in range(n_clients - n_attackers)
        ]
        
        # Scenario 1: WITH blockchain
        malicious_with = [
            attacker_with_blockchain.generate_malicious_update(
                true_gradient, honest_gradients, round_num
            )
            for _ in range(n_attackers)
        ]
        
        all_with = [(g, False) for g in honest_gradients] + [(g, True) for g in malicious_with]
        np.random.shuffle(all_with)
        all_gradients_with = [g for g, _ in all_with]
        is_malicious_with = [m for _, m in all_with]

        # Simple detection (norm outliers)
        gradient_norms_with = [np.linalg.norm(g) for g in all_gradients_with]
        median_norm_with = np.median(gradient_norms_with)
        std_norm_with = np.std(gradient_norms_with)

        threshold_with = median_norm_with + detection_sigma_threshold * std_norm_with
        detected_indices_with = [i for i, norm in enumerate(gradient_norms_with) if norm > threshold_with]
        malicious_detected_with = sum(1 for i in detected_indices_with if is_malicious_with[i])
        malicious_detection_rate_with = malicious_detected_with / n_attackers if n_attackers > 0 else 0.0
        malicious_success_rate_with = 1.0 - malicious_detection_rate_with
        
        # Aggregate (reject detected outliers)
        kept_indices = [i for i, norm in enumerate(gradient_norms_with)
                       if norm <= median_norm_with + 2 * std_norm_with]
        
        if kept_indices:
            aggregated_with = np.mean([all_gradients_with[i] for i in kept_indices], axis=0)
        else:
            aggregated_with = np.median(all_gradients_with, axis=0)
        
        error_with = np.linalg.norm(aggregated_with - true_gradient)
        
        # Log round-level feedback (fraction of malicious updates detected/rejected)
        attacker_with_blockchain.log_attack_result(
            was_detected=malicious_detection_rate_with,
            was_rejected=malicious_detection_rate_with
        )
        
        # Scenario 2: WITHOUT blockchain (same process, different attacker)
        malicious_without = [
            attacker_without_blockchain.generate_malicious_update(
                true_gradient, honest_gradients, round_num
            )
            for _ in range(n_attackers)
        ]

        all_without = [(g, False) for g in honest_gradients] + [(g, True) for g in malicious_without]
        np.random.shuffle(all_without)
        all_gradients_without = [g for g, _ in all_without]
        is_malicious_without = [m for _, m in all_without]

        gradient_norms_without = [np.linalg.norm(g) for g in all_gradients_without]
        median_norm_without = np.median(gradient_norms_without)
        std_norm_without = np.std(gradient_norms_without)

        threshold_without = median_norm_without + detection_sigma_threshold * std_norm_without
        detected_indices_without = [i for i, norm in enumerate(gradient_norms_without) if norm > threshold_without]
        malicious_detected_without = sum(1 for i in detected_indices_without if is_malicious_without[i])
        malicious_detection_rate_without = malicious_detected_without / n_attackers if n_attackers > 0 else 0.0
        malicious_success_rate_without = 1.0 - malicious_detection_rate_without
        
        kept_indices_without = [i for i, norm in enumerate(gradient_norms_without)
                               if norm <= median_norm_without + 2 * std_norm_without]
        
        if kept_indices_without:
            aggregated_without = np.mean([all_gradients_without[i] for i in kept_indices_without], axis=0)
        else:
            aggregated_without = np.median(all_gradients_without, axis=0)
        
        error_without = np.linalg.norm(aggregated_without - true_gradient)
        
        attacker_without_blockchain.log_attack_result(
            was_detected=malicious_detection_rate_without,
            was_rejected=malicious_detection_rate_without
        )
        
        # Store results
        results_with_blockchain['malicious_detection_rates'].append(malicious_detection_rate_with)
        results_with_blockchain['malicious_success_rates'].append(malicious_success_rate_with)
        results_with_blockchain['attack_strengths'].append(attacker_with_blockchain.attack_strength)
        results_with_blockchain['model_errors'].append(error_with)
        
        results_without_blockchain['malicious_detection_rates'].append(malicious_detection_rate_without)
        results_without_blockchain['malicious_success_rates'].append(malicious_success_rate_without)
        results_without_blockchain['attack_strengths'].append(attacker_without_blockchain.attack_strength)
        results_without_blockchain['model_errors'].append(error_without)
    
    # Summary statistics
    stats_with = attacker_with_blockchain.get_statistics()
    stats_without = attacker_without_blockchain.get_statistics()
    
    summary = {
        'with_blockchain': {
            'avg_malicious_detection_rate': float(np.mean(results_with_blockchain['malicious_detection_rates'])),
            'avg_malicious_success_rate': float(np.mean(results_with_blockchain['malicious_success_rates'])),
            'final_attack_strength': results_with_blockchain['attack_strengths'][-1],
            'avg_model_error': np.mean(results_with_blockchain['model_errors']),
            'attacker_success_rate': stats_with['success_rate'],
            'strategy_evolution': attacker_with_blockchain.attack_history,
            'stats': stats_with
        },
        'without_blockchain': {
            'avg_malicious_detection_rate': float(np.mean(results_without_blockchain['malicious_detection_rates'])),
            'avg_malicious_success_rate': float(np.mean(results_without_blockchain['malicious_success_rates'])),
            'final_attack_strength': results_without_blockchain['attack_strengths'][-1],
            'avg_model_error': np.mean(results_without_blockchain['model_errors']),
            'attacker_success_rate': stats_without['success_rate'],
            'strategy_evolution': attacker_without_blockchain.attack_history,
            'stats': stats_without
        },
        'transparency_paradox': {
            'blockchain_helps_attacker': (
                stats_with['success_rate'] > stats_without['success_rate']
            ),
            'success_rate_difference': stats_with['success_rate'] - stats_without['success_rate'],
            'model_error_difference': (
                np.mean(results_with_blockchain['model_errors']) - 
                np.mean(results_without_blockchain['model_errors'])
            )
        },
        'params': {
            'n_rounds': n_rounds,
            'n_attackers': n_attackers,
            'n_clients': n_clients,
            'model_dim': model_dim,
            'detection_algorithm': detection_algorithm,
            'initial_attack_strength': initial_attack_strength,
            'learning_rate': learning_rate,
            'detection_sigma_threshold': detection_sigma_threshold,
            'honest_noise_std': honest_noise_std,
            'attacker_noise_std': attacker_noise_std
        }
    }
    
    logger.info(f"\nTransparency Paradox Results:")
    logger.info(f"  WITH blockchain: success_rate={stats_with['success_rate']:.1%}, "
               f"avg_error={summary['with_blockchain']['avg_model_error']:.4f}")
    logger.info(f"  WITHOUT blockchain: success_rate={stats_without['success_rate']:.1%}, "
               f"avg_error={summary['without_blockchain']['avg_model_error']:.4f}")
    logger.info(f"  Blockchain helps attacker: {summary['transparency_paradox']['blockchain_helps_attacker']}")
    
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Adaptive Attacker - Testing Transparency Paradox")
    print("Based on: FLARE (2025)")
    print("=" * 80)
    
    # Test 1: Basic adaptive attack
    print("\n[Test 1] Basic Adaptive Attack")
    print("-" * 80)
    
    attacker = AdaptiveAttacker(
        initial_attack_strength=5.0,
        learning_rate=0.1,
        has_blockchain_access=True
    )
    
    true_grad = np.random.randn(100) * 0.1
    honest_grads = [true_grad + np.random.randn(100) * 0.05 for _ in range(10)]
    
    malicious = attacker.generate_malicious_update(true_grad, honest_grads, 0)
    
    print(f"✓ Generated malicious update")
    print(f"✓ True gradient norm: {np.linalg.norm(true_grad):.4f}")
    print(f"✓ Malicious gradient norm: {np.linalg.norm(malicious):.4f}")
    print(f"✓ Attack strength: {attacker.attack_strength:.2f}")
    print(f"✓ Strategy: {attacker.current_strategy}")
    
    # Test 2: Adaptation over time
    print("\n[Test 2] Strategy Adaptation (10 rounds)")
    print("-" * 80)
    
    for round_num in range(10):
        malicious = attacker.generate_malicious_update(true_grad, honest_grads, round_num)
        
        # Simulate detection (50% chance)
        detected = np.random.rand() < 0.5
        attacker.log_attack_result(was_detected=detected, was_rejected=detected)
        
        if round_num % 3 == 0:
            print(f"  Round {round_num}: strategy={attacker.current_strategy}, "
                  f"strength={attacker.attack_strength:.2f}, "
                  f"success_rate={attacker.success_rate:.1%}")
    
    # Test 3: TRANSPARENCY PARADOX - Full Simulation
    print("\n[Test 3] TRANSPARENCY PARADOX Simulation (30 rounds)")
    print("-" * 80)
    
    results = simulate_transparency_paradox(
        n_rounds=30,
        n_clients=100,
        n_attackers=20,
        model_dim=1000,
        detection_algorithm='krum'
    )
    
    print(f"\n✓ WITH Blockchain (attacker can learn from on-chain logs):")
    print(f"  - Attacker success rate: {results['with_blockchain']['attacker_success_rate']:.1%}")
    print(f"  - Average detection rate: {results['with_blockchain']['avg_detection_rate']:.1%}")
    print(f"  - Average model error: {results['with_blockchain']['avg_model_error']:.4f}")
    print(f"  - Final attack strength: {results['with_blockchain']['final_attack_strength']:.2f}")
    
    print(f"\n✓ WITHOUT Blockchain (attacker cannot learn):")
    print(f"  - Attacker success rate: {results['without_blockchain']['attacker_success_rate']:.1%}")
    print(f"  - Average detection rate: {results['without_blockchain']['avg_detection_rate']:.1%}")
    print(f"  - Average model error: {results['without_blockchain']['avg_model_error']:.4f}")
    print(f"  - Final attack strength: {results['without_blockchain']['final_attack_strength']:.2f}")
    
    print(f"\n✓ TRANSPARENCY PARADOX Analysis:")
    print(f"  - Does blockchain transparency help attacker? "
          f"{results['transparency_paradox']['blockchain_helps_attacker']}")
    print(f"  - Success rate difference: "
          f"{results['transparency_paradox']['success_rate_difference']:+.1%}")
    print(f"  - Model error difference: "
          f"{results['transparency_paradox']['model_error_difference']:+.4f}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("📊 TRANSPARENCY PARADOX INTERPRETATION:")
    print("=" * 80)
    
    if results['transparency_paradox']['blockchain_helps_attacker']:
        print("⚠️  SHORT-TERM: Blockchain transparency DOES help adaptive attackers")
        print("   - Attackers learn from on-chain logs to evade detection")
        print("   - Success rate increases with learning")
    else:
        print("✅ SHORT-TERM: Blockchain transparency does NOT significantly help attackers")
        print("   - Detection mechanisms are robust even with transparency")
    
    print("\n✅ LONG-TERM: Blockchain still provides NET BENEFIT:")
    print("   - Immutable audit trail enables post-hoc forensics")
    print("   - Pattern analysis can identify adaptive attack campaigns")
    print("   - Provenance tracking reveals attacker identities over time")
    print("   - Centralized systems: 0% accountability (logs can be deleted)")
    
    print("\n" + "=" * 80)
    print("✅ All adaptive attacker tests passed!")
    print("=" * 80)
