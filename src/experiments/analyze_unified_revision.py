"""Generate statistics, tables, and figures from the unified actual experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "results" / "unified_mnist_actual.json"
SENSITIVITY_PATH = (
    ROOT / "results" / "unified_mnist_equal_weight_sensitivity.json"
)
ANALYSIS_PATH = ROOT / "results" / "unified_mnist_analysis.json"
VISUALIZATION_DIRECTORY = ROOT / "visualizations" / "revision_actual"
TABLE_DIRECTORY = ROOT / "visualizations" / "revision_actual"

METHOD_ORDER = [
    "clean_fedavg",
    "fedavg",
    "krum",
    "trimmed_mean",
    "atma",
]
METHOD_LABELS = {
    "clean_fedavg": "FedAvg (clean)",
    "fedavg": "FedAvg (attack)",
    "krum": "Krum",
    "trimmed_mean": "TrimmedMean",
    "atma": "MAD-ATMA",
}
COLORS = {
    "clean_fedavg": "#4d4d4d",
    "fedavg": "#d62728",
    "krum": "#ff7f0e",
    "trimmed_mean": "#2ca02c",
    "atma": "#1f77b4",
}


def load_results() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def load_sensitivity_results() -> dict:
    return json.loads(SENSITIVITY_PATH.read_text(encoding="utf-8"))


def runs_by_method(data: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in data["runs"]:
        grouped[run["method"]].append(run)
    for runs in grouped.values():
        runs.sort(key=lambda run: run["seed"])
    return grouped


def paired_tests(grouped: dict[str, list[dict]]) -> dict:
    comparisons = [
        ("clean_fedavg", "fedavg"),
        ("fedavg", "krum"),
        ("fedavg", "trimmed_mean"),
        ("fedavg", "atma"),
        ("trimmed_mean", "atma"),
    ]
    output = {}
    for first, second in comparisons:
        first_values = np.array(
            [run["final_accuracy"] * 100 for run in grouped[first]]
        )
        second_values = np.array(
            [run["final_accuracy"] * 100 for run in grouped[second]]
        )
        statistic, p_value = ttest_rel(first_values, second_values)
        differences = second_values - first_values
        output[f"{second}_minus_{first}"] = {
            "mean_difference_percentage_points": float(differences.mean()),
            "paired_t_statistic": float(statistic),
            "p_value_two_sided": float(p_value),
            "seedwise_differences_percentage_points": [
                float(value) for value in differences
            ],
        }
    return output


def weighting_sensitivity(
    grouped: dict[str, list[dict]],
    sensitivity_grouped: dict[str, list[dict]],
) -> dict:
    weighted = np.array(
        [run["final_accuracy"] * 100 for run in grouped["fedavg"]]
    )
    equal = np.array(
        [
            run["final_accuracy"] * 100
            for run in sensitivity_grouped["fedavg_equal"]
        ]
    )
    statistic, p_value = ttest_rel(weighted, equal)
    output = {
        "source_result": str(SENSITIVITY_PATH.relative_to(ROOT)),
        "summary": load_sensitivity_results()["summary"]["fedavg_equal"],
        "equal_minus_sample_weighted_fedavg": {
            "mean_difference_percentage_points": float((equal - weighted).mean()),
            "paired_t_statistic": float(statistic),
            "p_value_two_sided": float(p_value),
            "seedwise_differences_percentage_points": [
                float(value) for value in equal - weighted
            ],
        },
        "robust_method_minus_equal_client_mean": {},
    }
    for method in ("krum", "trimmed_mean", "atma"):
        values = np.array(
            [run["final_accuracy"] * 100 for run in grouped[method]]
        )
        method_statistic, method_p_value = ttest_rel(equal, values)
        output["robust_method_minus_equal_client_mean"][method] = {
            "mean_difference_percentage_points": float((values - equal).mean()),
            "paired_t_statistic": float(method_statistic),
            "p_value_two_sided": float(method_p_value),
        }
    return output


def atma_detection(grouped: dict[str, list[dict]]) -> dict:
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    trim_ratios = []
    flagged_counts = []
    for run in grouped["atma"]:
        for round_result in run["rounds"]:
            detection = round_result["metadata"]["detection"]
            for key in totals:
                totals[key] += int(detection[key])
            trim_ratios.append(float(round_result["metadata"]["trim_ratio"]))
            flagged_counts.append(
                len(round_result["metadata"]["flagged_client_ids"])
            )

    precision = totals["tp"] / (totals["tp"] + totals["fp"])
    recall = totals["tp"] / (totals["tp"] + totals["fn"])
    f1 = 2 * precision * recall / (precision + recall)
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rounds_evaluated": len(trim_ratios),
        "trim_ratio_min": min(trim_ratios),
        "trim_ratio_max": max(trim_ratios),
        "trim_ratio_final_by_seed": [
            float(run["rounds"][-1]["metadata"]["trim_ratio"])
            for run in grouped["atma"]
        ],
        "flagged_count_min": min(flagged_counts),
        "flagged_count_max": max(flagged_counts),
    }


def krum_selection(grouped: dict[str, list[dict]]) -> dict:
    selected_byzantine = 0
    selected_honest = 0
    for run in grouped["krum"]:
        byzantine_ids = set(run["byzantine_client_ids"])
        for round_result in run["rounds"]:
            selected = round_result["metadata"]["selected_client_id"]
            if selected in byzantine_ids:
                selected_byzantine += 1
            else:
                selected_honest += 1
    return {
        "selected_byzantine_rounds": selected_byzantine,
        "selected_honest_rounds": selected_honest,
        "rounds_evaluated": selected_byzantine + selected_honest,
    }


def blockchain_summary(data: dict) -> dict:
    evidence = data["blockchain_evidence"]
    operations: dict[str, list[int]] = defaultdict(list)
    for transaction in evidence["transactions"]:
        operations[transaction["operation"]].append(transaction["gas_used"])
    return {
        "chain_id": evidence["chain_id"],
        "contract_address": evidence["contract_address"],
        "deployment_tx_hash": evidence["deployment"]["tx_hash"],
        "deployment_block_number": evidence["deployment"]["block_number"],
        "deployment_gas_used": evidence["deployment"]["gas_used"],
        "latest_block": evidence["latest_block"],
        "transaction_count_excluding_deployment": evidence["transaction_count"],
        "successful_transaction_count": sum(
            transaction["status"] == 1
            for transaction in evidence["transactions"]
        ),
        "total_gas_used_including_deployment": evidence[
            "total_gas_used_including_deployment"
        ],
        "operation_gas": {
            operation: {
                "count": len(values),
                "total": int(sum(values)),
                "mean": float(np.mean(values)),
                "min": int(min(values)),
                "max": int(max(values)),
            }
            for operation, values in operations.items()
        },
        "all_readback_verifications_passed": all(
            round_result["blockchain"]["verification_passed"]
            for run in data["runs"]
            if run["method"] == "atma" and run["seed"] == 42
            for round_result in run["rounds"]
        ),
        "all_client_readback_verifications_passed": all(
            round_result["blockchain"]["client_verification_passed"]
            for run in data["runs"]
            if run["method"] == "atma" and run["seed"] == 42
            for round_result in run["rounds"]
        ),
        "all_summary_readback_verifications_passed": all(
            round_result["blockchain"]["summary_verification_passed"]
            for run in data["runs"]
            if run["method"] == "atma" and run["seed"] == 42
            for round_result in run["rounds"]
        ),
    }


def plot_convergence(grouped: dict[str, list[dict]]) -> None:
    plt.figure(figsize=(3.35, 2.55))
    rounds = np.arange(1, len(grouped["atma"][0]["rounds"]) + 1)
    for method in METHOD_ORDER:
        matrix = np.array(
            [
                [
                    round_result["accuracy"] * 100
                    for round_result in run["rounds"]
                ]
                for run in grouped[method]
            ]
        )
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        plt.plot(
            rounds,
            mean,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            linewidth=1.4,
            marker="o",
            markersize=2.3,
            markevery=2,
        )
        plt.fill_between(
            rounds,
            mean - std,
            mean + std,
            color=COLORS[method],
            alpha=0.12,
        )
    plt.xlabel("Communication round", fontsize=8)
    plt.ylabel("Test accuracy (%)", fontsize=8)
    plt.xticks(rounds[::2])
    plt.ylim(0, 90)
    plt.grid(alpha=0.25)
    plt.tick_params(axis="both", labelsize=7)
    plt.legend(ncol=2, frameon=True, fontsize=6.4)
    plt.tight_layout(pad=0.4)
    plt.savefig(
        VISUALIZATION_DIRECTORY / "figure1_convergence_actual.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_final_accuracy(data: dict) -> None:
    means = [
        data["summary"][method]["mean_accuracy_percent"]
        for method in METHOD_ORDER
    ]
    margins = [
        data["summary"][method]["confidence_interval_95_margin_percent"]
        for method in METHOD_ORDER
    ]
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    colors = [COLORS[method] for method in METHOD_ORDER]

    plt.figure(figsize=(3.35, 2.45))
    positions = np.arange(len(labels))
    bars = plt.bar(
        positions,
        means,
        yerr=margins,
        capsize=3,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    plt.xticks(positions, labels, rotation=18, ha="right", fontsize=6.2)
    plt.yticks(fontsize=7)
    plt.ylabel("Final accuracy (%)", fontsize=8)
    plt.ylim(0, 95)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout(pad=0.4)
    plt.savefig(
        VISUALIZATION_DIRECTORY / "figure2_final_accuracy_actual.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def write_results_table(data: dict, sensitivity_data: dict) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Final MNIST Accuracy Across Three Seeds}",
        r"\label{tab:actual_results}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Mean (\%) & SD & 95\% CI \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        summary = data["summary"][method]
        lower, upper = summary["confidence_interval_95_percent"]
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{summary['mean_accuracy_percent']:.2f} & "
            f"{summary['sample_std_percent']:.2f} & "
            f"[{lower:.2f}, {upper:.2f}] \\\\"
        )
        if method == "fedavg":
            sensitivity = sensitivity_data["summary"]["fedavg_equal"]
            sensitivity_lower, sensitivity_upper = sensitivity[
                "confidence_interval_95_percent"
            ]
            lines.append(
                "Equal-client mean (attack) & "
                f"{sensitivity['mean_accuracy_percent']:.2f} & "
                f"{sensitivity['sample_std_percent']:.2f} & "
                f"[{sensitivity_lower:.2f}, {sensitivity_upper:.2f}] \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (TABLE_DIRECTORY / "actual_results_table.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_blockchain_table(blockchain: dict) -> None:
    client = blockchain["operation_gas"]["record_client_update"]
    rounds = blockchain["operation_gas"]["finalize_round"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Measured Ganache Audit Transactions}",
        r"\label{tab:blockchain_actual}",
        r"\centering",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Operation & Count & Mean Gas \\",
        r"\midrule",
        f"Contract deployment & 1 & "
        f"{blockchain['deployment_gas_used']:,} \\\\",
        f"Client update record & {client['count']} & "
        f"{client['mean']:,.0f} \\\\",
        f"Round finalization & {rounds['count']} & "
        f"{rounds['mean']:,.0f} \\\\",
        r"\midrule",
        f"Total measured gas & -- & "
        f"{blockchain['total_gas_used_including_deployment']:,} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (TABLE_DIRECTORY / "blockchain_actual_table.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    data = load_results()
    sensitivity_data = load_sensitivity_results()
    grouped = runs_by_method(data)
    sensitivity_grouped = runs_by_method(sensitivity_data)
    analysis = {
        "source_result": str(RESULT_PATH.relative_to(ROOT)),
        "summary": data["summary"],
        "paired_tests": paired_tests(grouped),
        "weighting_sensitivity": weighting_sensitivity(
            grouped,
            sensitivity_grouped,
        ),
        "atma_detection": atma_detection(grouped),
        "krum_selection": krum_selection(grouped),
        "blockchain": blockchain_summary(data),
    }

    VISUALIZATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )
    plot_convergence(grouped)
    plot_final_accuracy(data)
    write_results_table(data, sensitivity_data)
    write_blockchain_table(analysis["blockchain"])
    print(f"Saved analysis: {ANALYSIS_PATH}")
    print(f"Saved figures and tables: {VISUALIZATION_DIRECTORY}")


if __name__ == "__main__":
    main()
