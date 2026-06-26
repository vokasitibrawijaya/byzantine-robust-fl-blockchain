"""Unified, reproducible experiment for the ETASR revision.

Every aggregation method uses the same dataset partition, model architecture,
client participation, local optimizer, attack, rounds, and random seed.
The optional ``fedavg_equal`` method is a client-equal sensitivity baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.aggregation.robust_aggregation import (
    AdaptiveTrimmedMean,
    TensorUpdate,
    coordinate_trimmed_mean,
    krum,
    mean_aggregate,
)
from src.blockchain.audit_chain import FLAuditChain, hash_tensors


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_FILE = REPOSITORY_ROOT / "results" / "unified_mnist_actual.json"
DEFAULT_ARTIFACT_DIRECTORY = (
    REPOSITORY_ROOT / "artifacts" / "blockchain" / "fl_audit"
)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "MNIST"
    model: str = "SmallMNISTCNN"
    total_clients: int = 20
    clients_per_round: int = 20
    byzantine_clients: int = 4
    dirichlet_alpha: float = 0.5
    rounds: int = 20
    local_epochs: int = 1
    max_batches_per_client: int = 3
    batch_size: int = 64
    learning_rate: float = 0.05
    attack: str = "label_flip_plus_scaled_model_delta"
    attack_scale: float = 5.0
    label_shift: int = 1
    static_trim_count_each_tail: int = 4
    seeds: tuple[int, ...] = (42, 43, 44)


class SmallMNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.pool(F.relu(self.conv1(inputs)))
        inputs = self.pool(F.relu(self.conv2(inputs)))
        inputs = inputs.flatten(start_dim=1)
        inputs = F.relu(self.fc1(inputs))
        return self.fc2(inputs)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mnist() -> tuple:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_directory = REPOSITORY_ROOT / "data"
    training = datasets.MNIST(
        data_directory,
        train=True,
        download=True,
        transform=transform,
    )
    testing = datasets.MNIST(
        data_directory,
        train=False,
        download=True,
        transform=transform,
    )
    return training, testing


def dirichlet_partition(
    labels: np.ndarray,
    client_count: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    generator = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(client_count)]
    for class_id in sorted(np.unique(labels)):
        class_indices = np.where(labels == class_id)[0]
        generator.shuffle(class_indices)
        proportions = generator.dirichlet(np.full(client_count, alpha))
        boundaries = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(class_indices, boundaries)):
            client_indices[client_id].extend(int(index) for index in split)
    for indices in client_indices:
        generator.shuffle(indices)
    if any(not indices for indices in client_indices):
        raise RuntimeError("Dirichlet partition produced an empty client")
    return client_indices


def parameter_tensors(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def model_delta(
    global_model: nn.Module,
    local_model: nn.Module,
) -> TensorUpdate:
    return [
        local.detach() - global_parameter.detach()
        for global_parameter, local in zip(
            global_model.parameters(),
            local_model.parameters(),
        )
    ]


def train_client(
    global_model: nn.Module,
    training_dataset,
    indices: list[int],
    client_id: int,
    round_number: int,
    seed: int,
    config: ExperimentConfig,
    device: torch.device,
    attack_enabled: bool,
    attack_scale: float | None = None,
) -> tuple[TensorUpdate, float]:
    local_model = SmallMNISTCNN().to(device)
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        seed * 1_000_000 + round_number * 1_000 + client_id
    )
    loader = DataLoader(
        Subset(training_dataset, indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=config.learning_rate,
    )
    loss_function = nn.CrossEntropyLoss()
    losses: list[float] = []

    for _ in range(config.local_epochs):
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index >= config.max_batches_per_client:
                break
            images = images.to(device)
            labels = labels.to(device)
            if attack_enabled:
                labels = (labels + config.label_shift) % 10

            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(local_model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

    delta = model_delta(global_model, local_model)
    if attack_enabled:
        scale = config.attack_scale if attack_scale is None else attack_scale
        delta = [tensor * scale for tensor in delta]
    return delta, float(np.mean(losses))


def apply_update(model: nn.Module, update: TensorUpdate) -> None:
    with torch.no_grad():
        for parameter, delta in zip(model.parameters(), update):
            parameter.add_(delta)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    sample_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += float(loss_function(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            sample_count += int(labels.numel())
    return correct / sample_count, total_loss / sample_count


def detection_metrics(
    flagged_ids: set[int],
    byzantine_ids: set[int],
    client_count: int,
) -> dict:
    all_ids = set(range(client_count))
    honest_ids = all_ids - byzantine_ids
    tp = len(flagged_ids & byzantine_ids)
    fp = len(flagged_ids & honest_ids)
    fn = len(byzantine_ids - flagged_ids)
    tn = len(honest_ids - flagged_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_method(
    method: str,
    seed: int,
    config: ExperimentConfig,
    training_dataset,
    testing_loader: DataLoader,
    partitions: list[list[int]],
    device: torch.device,
    audit_chain: FLAuditChain | None = None,
) -> dict:
    set_determinism(seed)
    model = SmallMNISTCNN().to(device)
    byzantine_ids = set(range(config.byzantine_clients))
    attack_enabled = method != "clean_fedavg"
    atma = AdaptiveTrimmedMean(
        initial_trim_ratio=0.10,
        min_trim_ratio=0.05,
        max_trim_ratio=0.25,
        adaptation_rate=0.50,
        outlier_z_threshold=3.5,
    )
    rounds: list[dict] = []
    start_time = time.perf_counter()

    for round_number in range(1, config.rounds + 1):
        updates: list[TensorUpdate] = []
        sample_weights: list[int] = []
        local_losses: list[float] = []
        for client_id in range(config.clients_per_round):
            is_byzantine = attack_enabled and client_id in byzantine_ids
            update, local_loss = train_client(
                model,
                training_dataset,
                partitions[client_id],
                client_id,
                round_number,
                seed,
                config,
                device,
                is_byzantine,
            )
            updates.append(update)
            sample_weights.append(len(partitions[client_id]))
            local_losses.append(local_loss)

        metadata: dict = {}
        if method in {"clean_fedavg", "fedavg"}:
            aggregate = mean_aggregate(updates, weights=sample_weights)
        elif method == "fedavg_equal":
            aggregate = mean_aggregate(updates)
        elif method == "krum":
            aggregate, metadata = krum(
                updates,
                byzantine_bound=config.byzantine_clients,
            )
            metadata["selected_client_id"] = metadata.pop(
                "selected_client_position"
            )
        elif method == "trimmed_mean":
            aggregate = coordinate_trimmed_mean(
                updates,
                trim_count=config.static_trim_count_each_tail,
            )
        elif method == "atma":
            aggregate, metadata = atma.aggregate(
                updates,
                client_ids=list(range(config.clients_per_round)),
            )
            flagged = set(metadata["flagged_client_ids"])
            metadata["detection"] = detection_metrics(
                flagged,
                byzantine_ids,
                config.clients_per_round,
            )
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        apply_update(model, aggregate)
        accuracy, test_loss = evaluate(model, testing_loader, device)

        blockchain_record = None
        if audit_chain is not None:
            flagged_ids = set(metadata.get("flagged_client_ids", []))
            client_verification_passed = True
            for client_id, update in enumerate(updates):
                update_hash = hash_tensors(update)
                audit_chain.record_client_update(
                    round_number,
                    client_id,
                    update_hash,
                    client_id in flagged_ids,
                )
                client_verification_passed &= audit_chain.verify_client_update(
                    round_number,
                    client_id,
                    update_hash,
                    client_id in flagged_ids,
                )
            aggregate_hash = hash_tensors(aggregate)
            receipt = audit_chain.finalize_round(
                round_number,
                aggregate_hash,
                config.clients_per_round,
                len(flagged_ids),
                metadata["trim_count_each_tail"],
            )
            summary_verification_passed = audit_chain.verify_round_summary(
                round_number,
                aggregate_hash,
                config.clients_per_round,
                len(flagged_ids),
                metadata["trim_count_each_tail"],
            )
            blockchain_record = {
                "aggregate_hash": "0x" + aggregate_hash.hex(),
                "client_verification_passed": client_verification_passed,
                "summary_verification_passed": summary_verification_passed,
                "verification_passed": (
                    client_verification_passed
                    and summary_verification_passed
                ),
                "finalize_tx_hash": receipt["tx_hash"],
                "finalize_block_number": receipt["block_number"],
            }

        round_result = {
            "round": round_number,
            "accuracy": accuracy,
            "test_loss": test_loss,
            "mean_local_loss": float(np.mean(local_losses)),
            "metadata": metadata,
            "blockchain": blockchain_record,
        }
        rounds.append(round_result)
        print(
            f"{method:13s} seed={seed} round={round_number:02d}/"
            f"{config.rounds} accuracy={accuracy * 100:6.2f}%"
        )

    return {
        "method": method,
        "seed": seed,
        "attack_enabled": attack_enabled,
        "byzantine_client_ids": sorted(byzantine_ids) if attack_enabled else [],
        "final_accuracy": rounds[-1]["accuracy"],
        "final_test_loss": rounds[-1]["test_loss"],
        "runtime_seconds": time.perf_counter() - start_time,
        "rounds": rounds,
    }


def summarize(runs: list[dict]) -> dict:
    summaries: dict[str, dict] = {}
    methods = sorted({run["method"] for run in runs})
    for method in methods:
        values = np.array(
            [
                run["final_accuracy"] * 100
                for run in runs
                if run["method"] == method
            ],
            dtype=float,
        )
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if len(values) > 1:
            margin = float(t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
        else:
            margin = 0.0
        summaries[method] = {
            "n": int(len(values)),
            "accuracies_percent": [float(value) for value in values],
            "mean_accuracy_percent": mean,
            "sample_std_percent": std,
            "confidence_interval_95_percent": [mean - margin, mean + margin],
            "confidence_interval_95_margin_percent": margin,
        }
    return summaries


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["clean_fedavg", "fedavg", "krum", "trimmed_mean", "atma"],
    )
    parser.add_argument("--blockchain", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_FILE)
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
    runs: list[dict] = []
    blockchain_evidence = None
    audit_chain = None

    if arguments.blockchain:
        audit_chain = FLAuditChain(DEFAULT_ARTIFACT_DIRECTORY)

    for seed in config.seeds:
        partitions = dirichlet_partition(
            labels,
            config.total_clients,
            config.dirichlet_alpha,
            seed,
        )
        for method in arguments.methods:
            method_chain = (
                audit_chain
                if audit_chain is not None and method == "atma" and seed == 42
                else None
            )
            runs.append(
                run_method(
                    method,
                    seed,
                    config,
                    training_dataset,
                    testing_loader,
                    partitions,
                    device,
                    audit_chain=method_chain,
                )
            )

    if audit_chain is not None:
        blockchain_evidence = audit_chain.evidence()

    output = {
        "schema_version": 1,
        "created_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "runs": runs,
        "summary": summarize(runs),
        "blockchain_evidence": blockchain_evidence,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {arguments.output}")


if __name__ == "__main__":
    main()
