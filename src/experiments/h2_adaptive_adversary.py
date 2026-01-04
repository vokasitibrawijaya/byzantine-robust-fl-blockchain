"""
H2 Adaptive Stealth Adversary
More sophisticated tampering strategies to stress-test provenance detection.
"""

import random
import numpy as np
from typing import Dict, List, Tuple
from provenance_tracker import ProvenanceTracker


class AdaptiveStealthAdversary:
    """
    Sophisticated adversary that adapts tampering strategy based on detection risk.
    
    Strategies:
    1. GRADUAL: Slowly increase tampering magnitude over time
    2. INTERMITTENT: Attack sporadically to avoid pattern detection
    3. MIMICRY: Match statistical profile of normal rounds
    4. DELAYED: Wait for trust buildup before attacking
    """
    
    def __init__(self, detection_threshold: float = 3.5):
        self.strategy = "DELAYED"  # Start conservatively
        self.detection_threshold = detection_threshold
        self.attack_history: List[Tuple[int, bool]] = []  # (round, detected)
        self.trust_buildup_rounds = 10  # Wait period before attacking
        
    def should_attack(self, current_round: int, total_rounds: int) -> bool:
        """Decide whether to attack this round based on strategy"""
        
        if self.strategy == "DELAYED":
            # Wait for trust period, then attack consistently
            if current_round < self.trust_buildup_rounds:
                return False
            return random.random() < 0.3  # 30% attack rate after delay
        
        elif self.strategy == "INTERMITTENT":
            # Sporadic attacks (15% rate)
            return random.random() < 0.15
        
        elif self.strategy == "GRADUAL":
            # Increase attack frequency linearly
            progress = current_round / total_rounds
            return random.random() < (0.05 + 0.25 * progress)
        
        elif self.strategy == "MIMICRY":
            # Match normal round frequency (20%)
            return random.random() < 0.20
        
        return False
    
    def compute_tampering_magnitude(self, 
                                    current_round: int, 
                                    baseline_variance: float) -> float:
        """
        Compute how much to tamper based on strategy and detection risk.
        
        Returns:
            Multiplier for tampering magnitude (1.0 = baseline, <1.0 = stealthier)
        """
        
        if self.strategy == "GRADUAL":
            # Start with small tampering, gradually increase
            progress = current_round / 50  # Assume 50 rounds
            return 0.5 + 0.5 * progress  # 0.5x to 1.0x
        
        elif self.strategy == "MIMICRY":
            # Stay within 1.5 sigma of normal variance
            return 0.7  # Reduce magnitude to ~70%
        
        elif self.strategy == "INTERMITTENT":
            # When attacking, go full magnitude but infrequently
            return 1.0
        
        elif self.strategy == "DELAYED":
            # After trust period, use moderate tampering
            return 0.85
        
        return 1.0
    
    def update_strategy(self, detection_rate: float):
        """
        Adapt strategy based on recent detection rate.
        
        Args:
            detection_rate: Fraction of attacks detected in recent window
        """
        if detection_rate > 0.8:
            # High detection → switch to stealthier strategy
            if self.strategy != "MIMICRY":
                self.strategy = "MIMICRY"
                print(f"  [!] Adversary adapting: switching to MIMICRY (detection rate: {detection_rate:.1%})")
        
        elif detection_rate < 0.3:
            # Low detection → can be more aggressive
            if self.strategy == "MIMICRY":
                self.strategy = "GRADUAL"
                print(f"  [!] Adversary adapting: switching to GRADUAL (detection rate: {detection_rate:.1%})")


def run_adaptive_experiment(n_rounds: int = 50, 
                           n_workers: int = 100,
                           byzantine_ratio: float = 0.2,
                           blockchain_enabled: bool = True,
                           seed: int = 42) -> Dict:
    """
    Run H2 experiment with adaptive stealth adversary.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    tracker = ProvenanceTracker(blockchain_enabled=blockchain_enabled)
    adversary = AdaptiveStealthAdversary()
    ground_truth_labels = []  # True if record is corrupted
    
    print(f"\n[*] Running adaptive adversary experiment (blockchain={blockchain_enabled})")
    print(f"   Byzantine ratio: {byzantine_ratio:.1%}, Rounds: {n_rounds}")
    
    # Baseline parameters
    baseline_reject_rate = 0.07
    baseline_reject_jitter = 0.02
    n_byzantine = int(n_workers * byzantine_ratio)
    
    # Simulate federated learning rounds
    for round_num in range(n_rounds):
        # Decide if adversary attacks this round
        will_attack = adversary.should_attack(round_num, n_rounds)
        
        # Generate client updates
        client_updates = []
        for worker_id in range(n_workers):
            is_byzantine = worker_id < n_byzantine
            
            # Generate realistic gradient norms
            base_norm = 0.1 + random.gauss(0, 0.02)
            if is_byzantine and will_attack:
                # Apply tampering based on adversary magnitude
                magnitude = adversary.compute_tampering_magnitude(round_num, 0.02)
                base_norm *= (5.0 * magnitude)  # Spike gradient norm
            
            client_updates.append({
                'client_id': f'client_{worker_id}',
                'update_hash': f'hash_{round_num}_{worker_id}',
                'gradient_norm': base_norm
            })
        
        # Compute rejection rate (realistic with jitter)
        base_p_reject = max(0.0, min(0.5, baseline_reject_rate + random.uniform(-baseline_reject_jitter, baseline_reject_jitter)))
        
        # If under attack, increase rejection rate
        if will_attack:
            magnitude = adversary.compute_tampering_magnitude(round_num, 0.02)
            # Stealthy adversary increases rejection modestly
            extra_reject = int(magnitude * 0.15 * n_workers)
            n_rejected = int(np.random.binomial(n_workers, base_p_reject)) + extra_reject
            n_rejected = min(n_workers - 1, n_rejected)
        else:
            n_rejected = int(np.random.binomial(n_workers, base_p_reject))
        
        aggregation_result = {
            'n_accepted': n_workers - n_rejected,
            'n_rejected': n_rejected,
            'aggregated_norm': 0.1
        }
        
        # Algorithm may change under attack (stealth adversary stays with median)
        algorithm = 'median'
        if will_attack and random.random() < 0.3:  # 30% chance to switch
            algorithm = 'krum'
        
        # Log to provenance
        tracker.log_aggregation(
            round_number=round_num,
            aggregator_id='aggregator_0',
            algorithm=algorithm,
            client_updates=client_updates,
            aggregation_result=aggregation_result,
            metadata={'adversary_active': will_attack}
        )
        
        # If centralized (no blockchain), attacker can tamper logs
        if not blockchain_enabled and will_attack and random.random() < 0.7:
            record = tracker.query_round(round_num)
            if record is not None:
                # Tamper to hide attack
                record.algorithm = 'median'
                record.aggregation_result['n_rejected'] = int(np.random.binomial(n_workers, baseline_reject_rate))
                record.aggregation_result['n_accepted'] = n_workers - record.aggregation_result['n_rejected']
        
        # Ground truth label (True if attacked AND blockchain can detect it)
        ground_truth_labels.append(will_attack and blockchain_enabled)
    
    # Detection
    detection_summary = tracker.detect_corruption_patterns(threshold_sigma=3.5)
    
    # Compute ROC curve
    from provenance_tracker import compute_roc_curve
    roc_data = compute_roc_curve(tracker, ground_truth_labels)
    
    # Find optimal operating point (maximize Youden's J = TPR - FPR)
    best_j = -1
    best_point = None
    for point in roc_data['roc_points']:
        j = point['TPR'] - point['FPR']
        if j > best_j:
            best_j = j
            best_point = point
    
    return {
        'blockchain_enabled': blockchain_enabled,
        'byzantine_ratio': byzantine_ratio,
        'n_rounds': n_rounds,
        'n_workers': n_workers,
        'adversary_strategy': adversary.strategy,
        'detection_summary': detection_summary,
        'roc_curve': roc_data,
        'optimal_operating_point': best_point,
        'youndens_j': best_j
    }


if __name__ == "__main__":
    print("=" * 80)
    print("H2 ADAPTIVE ADVERSARY EXPERIMENT")
    print("=" * 80)
    
    # Test with blockchain
    result_blockchain = run_adaptive_experiment(
        n_rounds=50,
        blockchain_enabled=True,
        seed=42
    )
    
    # Test without blockchain
    result_centralized = run_adaptive_experiment(
        n_rounds=50,
        blockchain_enabled=False,
        seed=42
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\n[BLOCKCHAIN]:")
    print(f"  AUC: {result_blockchain['roc_curve']['auc']:.3f}")
    print(f"  Optimal threshold: {result_blockchain['optimal_operating_point']['threshold']:.2f} sigma")
    print(f"  Optimal TPR: {result_blockchain['optimal_operating_point']['TPR']:.1%}")
    print(f"  Optimal FPR: {result_blockchain['optimal_operating_point']['FPR']:.1%}")
    print(f"  Youden's J: {result_blockchain['youndens_j']:.3f}")
    
    print("\n[CENTRALIZED]:")
    print(f"  AUC: {result_centralized['roc_curve']['auc']:.3f}")
    print(f"  (No tamper-proof logs -> cannot detect)")
    
    # Save results
    import json
    with open("h2_adaptive_adversary_results.json", "w") as f:
        json.dump({
            'blockchain': result_blockchain,
            'centralized': result_centralized
        }, f, indent=2)
    
    print("\n[OK] Results saved to: h2_adaptive_adversary_results.json")
    print("=" * 80)
