"""Controlled ledger-information experiment for the Transparency Paradox."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from torch.utils.data import DataLoader

from src.aggregation.robust_aggregation import AdaptiveTrimmedMean
from src.blockchain.audit_chain import FLAuditChain, hash_tensors
from src.experiments.unified_mnist_blockchain_experiment import (
    DEFAULT_ARTIFACT_DIRECTORY,
    ExperimentConfig,
    SmallMNISTCNN,
    apply_update,
    detection_metrics,
    dirichlet_partition,
    evaluate,
    load_mnist,
    set_determinism,
    train_client,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "transparency_paradox_actual.json"
VIEW_CALLS_PER_FLAG_QUERY = 2


@dataclass
class LedgerFeedbackController:
    """Prespecified controller using only previous-round ledger flags."""

    initial_scale: float = 5.0
    minimum_scale: float = 1.0
    maximum_scale: float = 5.0
    high_detection_threshold: float = 0.75
    low_detection_threshold: float = 0.25
    high_detection_multiplier: float = 0.70
    medium_detection_multiplier: float = 0.90
    low_detection_multiplier: float = 1.10

    def __post_init__(self) -> None:
        self.scale = self.initial_scale

    def update(self, previous_detection_rate: float) -> float:
        if previous_detection_rate >= self.high_detection_threshold:
            multiplier = self.high_detection_multiplier
        elif previous_detection_rate <= self.low_detection_threshold:
            multiplier = self.low_detection_multiplier
        else:
            multiplier = self.medium_detection_multiplier
        self.scale = min(
            self.maximum_scale,
            max(self.minimum_scale, self.scale * multiplier),
        )
        return self.scale


def record_round(
    chain: FLAuditChain,
    round_number: int,
    updates,
    aggregate,
    metadata: dict,
    client_count: int,
) -> dict:
    flagged_ids = set(metadata["flagged_client_ids"])
    client_verification_passed = True
    for client_id, update in enumerate(updates):
        update_hash = hash_tensors(update)
        chain.record_client_update(
            round_number,
            client_id,
            update_hash,
            client_id in flagged_ids,
        )
        client_verification_passed &= chain.verify_client_update(
            round_number,
            client_id,
            update_hash,
            client_id in flagged_ids,
        )

    aggregate_hash = hash_tensors(aggregate)
    receipt = chain.finalize_round(
        round_number,
        aggregate_hash,
        client_count,
        len(flagged_ids),
        metadata["trim_count_each_tail"],
    )
    summary_verification_passed = chain.verify_round_summary(
        round_number,
        aggregate_hash,
        client_count,
        len(flagged_ids),
        metadata["trim_count_each_tail"],
    )
    return {
        "client_verification_passed": client_verification_passed,
        "summary_verification_passed": summary_verification_passed,
        "verification_passed": (
            client_verification_passed and summary_verification_passed
        ),
        "finalize_tx_hash": receipt["tx_hash"],
        "finalize_block_number": receipt["block_number"],
    }


def run_condition(
    condition: str,
    seed: int,
    config: ExperimentConfig,
    training_dataset,
    testing_loader: DataLoader,
    partitions: list[list[int]],
    device: torch.device,
) -> dict:
    if condition not in {"blind", "ledger_informed"}:
        raise ValueError(f"Unknown condition: {condition}")

    set_determinism(seed)
    model = SmallMNISTCNN().to(device)
    byzantine_ids = set(range(config.byzantine_clients))
    aggregator = AdaptiveTrimmedMean(
        initial_trim_ratio=0.10,
        min_trim_ratio=0.05,
        max_trim_ratio=0.25,
        adaptation_rate=0.50,
        outlier_z_threshold=3.5,
    )
    controller = LedgerFeedbackController(initial_scale=config.attack_scale)
    chain = FLAuditChain(DEFAULT_ARTIFACT_DIRECTORY)
    rounds = []
    ledger_flag_queries = 0
    ledger_contract_view_calls = 0
    start = time.perf_counter()

    for round_number in range(1, config.rounds + 1):
        previous_detection_rate = None
        if condition == "ledger_informed" and round_number > 1:
            previous_flags = [
                chain.read_client_flag(round_number - 1, client_id)
                for client_id in sorted(byzantine_ids)
            ]
            ledger_flag_queries += len(previous_flags)
            ledger_contract_view_calls += (
                len(previous_flags) * VIEW_CALLS_PER_FLAG_QUERY
            )
            previous_detection_rate = sum(previous_flags) / len(previous_flags)
            attack_scale = controller.update(previous_detection_rate)
        else:
            attack_scale = config.attack_scale

        updates = []
        local_losses = []
        for client_id in range(config.clients_per_round):
            update, local_loss = train_client(
                model,
                training_dataset,
                partitions[client_id],
                client_id,
                round_number,
                seed,
                config,
                device,
                client_id in byzantine_ids,
                attack_scale=attack_scale,
            )
            updates.append(update)
            local_losses.append(local_loss)

        aggregate, metadata = aggregator.aggregate(
            updates,
            client_ids=list(range(config.clients_per_round)),
        )
        flagged = set(metadata["flagged_client_ids"])
        metadata["detection"] = detection_metrics(
            flagged,
            byzantine_ids,
            config.clients_per_round,
        )
        metadata["attack_scale"] = attack_scale
        metadata["previous_ledger_detection_rate"] = previous_detection_rate

        apply_update(model, aggregate)
        accuracy, test_loss = evaluate(model, testing_loader, device)
        blockchain = record_round(
            chain,
            round_number,
            updates,
            aggregate,
            metadata,
            config.clients_per_round,
        )
        rounds.append(
            {
                "round": round_number,
                "accuracy": accuracy,
                "test_loss": test_loss,
                "mean_local_loss": float(np.mean(local_losses)),
                "metadata": metadata,
                "blockchain": blockchain,
            }
        )
        print(
            f"{condition:15s} seed={seed} round={round_number:02d}/"
            f"{config.rounds} scale={attack_scale:4.2f} "
            f"flags={len(flagged):02d} accuracy={accuracy * 100:6.2f}%"
        )

    evidence = chain.evidence()
    return {
        "condition": condition,
        "seed": seed,
        "byzantine_client_ids": sorted(byzantine_ids),
        "final_accuracy": rounds[-1]["accuracy"],
        "final_test_loss": rounds[-1]["test_loss"],
        "runtime_seconds": time.perf_counter() - start,
        "ledger_flag_queries": ledger_flag_queries,
        "ledger_contract_view_calls": ledger_contract_view_calls,
        "rounds": rounds,
        "blockchain_evidence": evidence,
    }


def summarize(runs: list[dict]) -> dict:
    output = {}
    for condition in ("blind", "ledger_informed"):
        selected = [run for run in runs if run["condition"] == condition]
        accuracies = np.array(
            [run["final_accuracy"] * 100 for run in selected],
            dtype=float,
        )
        std = float(accuracies.std(ddof=1))
        margin = float(t.ppf(0.975, len(accuracies) - 1) * std / math.sqrt(len(accuracies)))
        detections = [
            round_result["metadata"]["detection"]
            for run in selected
            for round_result in run["rounds"]
        ]
        tp = sum(item["tp"] for item in detections)
        fn = sum(item["fn"] for item in detections)
        scales = [
            round_result["metadata"]["attack_scale"]
            for run in selected
            for round_result in run["rounds"]
        ]
        output[condition] = {
            "n": len(selected),
            "final_accuracies_percent": accuracies.tolist(),
            "mean_final_accuracy_percent": float(accuracies.mean()),
            "sample_std_percent": std,
            "confidence_interval_95_percent": [
                float(accuracies.mean() - margin),
                float(accuracies.mean() + margin),
            ],
            "byzantine_detection_recall": tp / (tp + fn),
            "undetected_byzantine_rate": fn / (tp + fn),
            "mean_attack_scale": float(np.mean(scales)),
            "final_attack_scales": [
                run["rounds"][-1]["metadata"]["attack_scale"]
                for run in selected
            ],
            "ledger_flag_queries": sum(
                run["ledger_flag_queries"] for run in selected
            ),
            "ledger_contract_view_calls": sum(
                run["ledger_contract_view_calls"] for run in selected
            ),
            "all_readback_verifications_passed": all(
                round_result["blockchain"]["verification_passed"]
                for run in selected
                for round_result in run["rounds"]
            ),
            "client_record_transactions": sum(
                sum(
                    transaction["operation"] == "record_client_update"
                    for transaction in run["blockchain_evidence"]["transactions"]
                )
                for run in selected
            ),
            "round_summary_transactions": sum(
                sum(
                    transaction["operation"] == "finalize_round"
                    for transaction in run["blockchain_evidence"]["transactions"]
                )
                for run in selected
            ),
        }
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["blind", "ledger_informed"],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = ExperimentConfig(
        rounds=arguments.rounds,
        max_batches_per_client=arguments.max_batches,
        seeds=tuple(arguments.seeds),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_dataset, testing_dataset = load_mnist()
    testing_loader = DataLoader(
        testing_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    labels = np.asarray(training_dataset.targets)
    runs = []
    for seed in config.seeds:
        partitions = dirichlet_partition(
            labels,
            config.total_clients,
            config.dirichlet_alpha,
            seed,
        )
        for condition in arguments.conditions:
            runs.append(
                run_condition(
                    condition,
                    seed,
                    config,
                    training_dataset,
                    testing_loader,
                    partitions,
                    device,
                )
            )

    output = {
        "schema_version": 1,
        "created_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "controller": asdict(LedgerFeedbackController()),
        "runs": runs,
        "summary": summarize(runs),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved: {arguments.output}")


if __name__ == "__main__":
    main()
