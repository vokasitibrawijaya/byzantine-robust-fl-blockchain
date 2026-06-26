"""Analyze and visualize the actual Transparency Paradox experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "results" / "transparency_paradox_actual.json"
ANALYSIS_PATH = ROOT / "results" / "transparency_paradox_analysis.json"
OUTPUT_DIRECTORY = ROOT / "visualizations" / "revision_actual"


def selected_runs(data: dict, condition: str) -> list[dict]:
    return sorted(
        [run for run in data["runs"] if run["condition"] == condition],
        key=lambda run: run["seed"],
    )


def detection_recall(run: dict) -> float:
    detections = [
        round_result["metadata"]["detection"]
        for round_result in run["rounds"]
    ]
    tp = sum(item["tp"] for item in detections)
    fn = sum(item["fn"] for item in detections)
    return tp / (tp + fn)


def analyze(data: dict) -> dict:
    blind = selected_runs(data, "blind")
    informed = selected_runs(data, "ledger_informed")
    blind_accuracy = np.array(
        [run["final_accuracy"] * 100 for run in blind],
        dtype=float,
    )
    informed_accuracy = np.array(
        [run["final_accuracy"] * 100 for run in informed],
        dtype=float,
    )
    statistic, p_value = ttest_rel(informed_accuracy, blind_accuracy)
    blind_recall = np.array([detection_recall(run) for run in blind])
    informed_recall = np.array([detection_recall(run) for run in informed])

    return {
        "source_result": str(RESULT_PATH.relative_to(ROOT)),
        "final_accuracy": {
            "ledger_informed_minus_blind_percentage_points": float(
                (informed_accuracy - blind_accuracy).mean()
            ),
            "paired_t_statistic": float(statistic),
            "paired_p_value_two_sided": float(p_value),
            "seedwise_differences_percentage_points": [
                float(value)
                for value in informed_accuracy - blind_accuracy
            ],
        },
        "detection": {
            "blind_recall_by_seed": blind_recall.tolist(),
            "ledger_informed_recall_by_seed": informed_recall.tolist(),
            "mean_recall_reduction_percentage_points": float(
                (blind_recall - informed_recall).mean() * 100
            ),
            "ledger_informed_undetected_byzantine_updates": int(
                sum(
                    round_result["metadata"]["detection"]["fn"]
                    for run in informed
                    for round_result in run["rounds"]
                )
            ),
            "total_ledger_informed_byzantine_updates": int(
                sum(
                    round_result["metadata"]["detection"]["tp"]
                    + round_result["metadata"]["detection"]["fn"]
                    for run in informed
                    for round_result in run["rounds"]
                )
            ),
        },
        "blockchain": {
            "contracts_deployed": len(data["runs"]),
            "client_record_transactions": sum(
                data["summary"][condition]["client_record_transactions"]
                for condition in ("blind", "ledger_informed")
            ),
            "round_summary_transactions": sum(
                data["summary"][condition]["round_summary_transactions"]
                for condition in ("blind", "ledger_informed")
            ),
            "ledger_flag_queries": data["summary"]["ledger_informed"][
                "ledger_flag_queries"
            ],
            "ledger_contract_view_calls": data["summary"]["ledger_informed"][
                "ledger_contract_view_calls"
            ],
            "all_readback_verifications_passed": all(
                data["summary"][condition][
                    "all_readback_verifications_passed"
                ]
                for condition in ("blind", "ledger_informed")
            ),
            "total_gas_including_deployments": int(
                sum(
                    run["blockchain_evidence"][
                        "total_gas_used_including_deployment"
                    ]
                    for run in data["runs"]
                )
            ),
        },
    }


def write_table(data: dict) -> None:
    labels = {
        "blind": "Blind attacker",
        "ledger_informed": "Ledger-informed",
    }
    lines = [
        r"\begin{table}[t]",
        r"\caption{Transparency Paradox Results Across Three Seeds}",
        r"\label{tab:transparency_actual}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & Accuracy (\%) & Recall (\%) & Undetected (\%) & Scale \\",
        r"\midrule",
    ]
    for condition in ("blind", "ledger_informed"):
        summary = data["summary"][condition]
        lines.append(
            f"{labels[condition]} & "
            f"{summary['mean_final_accuracy_percent']:.2f} $\\pm$ "
            f"{summary['sample_std_percent']:.2f} & "
            f"{summary['byzantine_detection_recall'] * 100:.1f} & "
            f"{summary['undetected_byzantine_rate'] * 100:.1f} & "
            f"{summary['mean_attack_scale']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (OUTPUT_DIRECTORY / "transparency_actual_table.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plot(data: dict) -> None:
    blind = selected_runs(data, "blind")
    informed = selected_runs(data, "ledger_informed")
    rounds = np.arange(1, len(blind[0]["rounds"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25))

    accuracy_means = [
        data["summary"][condition]["mean_final_accuracy_percent"]
        for condition in ("blind", "ledger_informed")
    ]
    accuracy_sd = [
        data["summary"][condition]["sample_std_percent"]
        for condition in ("blind", "ledger_informed")
    ]
    axes[0].bar(
        [0, 1],
        accuracy_means,
        yerr=accuracy_sd,
        color=["#4d4d4d", "#d62728"],
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].set_xticks([0, 1], ["Blind", "Informed"])
    axes[0].set_ylabel("Final accuracy (%)")
    axes[0].set_ylim(70, 82)
    axes[0].set_title("(a) Utility")

    recalls = [
        data["summary"][condition]["byzantine_detection_recall"] * 100
        for condition in ("blind", "ledger_informed")
    ]
    axes[1].bar(
        [0, 1],
        recalls,
        color=["#2ca02c", "#ff7f0e"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].set_xticks([0, 1], ["Blind", "Informed"])
    axes[1].set_ylabel("Detection recall (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("(b) Evasion")

    scale_matrix = np.array(
        [
            [
                round_result["metadata"]["attack_scale"]
                for round_result in run["rounds"]
            ]
            for run in informed
        ]
    )
    axes[2].plot(
        rounds,
        scale_matrix.mean(axis=0),
        color="#1f77b4",
        marker="o",
        markersize=2.5,
        linewidth=1.3,
    )
    axes[2].fill_between(
        rounds,
        scale_matrix.min(axis=0),
        scale_matrix.max(axis=0),
        color="#1f77b4",
        alpha=0.15,
    )
    axes[2].axhline(5.0, color="#4d4d4d", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Attack scale")
    axes[2].set_xticks([1, 5, 10, 15, 20])
    axes[2].set_ylim(0.5, 5.5)
    axes[2].set_title("(c) Ledger adaptation")

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(labelsize=7)
        axis.title.set_fontsize(8)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)

    fig.tight_layout(pad=0.5, w_pad=0.8)
    fig.savefig(
        OUTPUT_DIRECTORY / "figure3_transparency_actual.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    analysis = analyze(data)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )
    write_table(data)
    plot(data)
    print(f"Saved: {ANALYSIS_PATH}")
    print(f"Saved table and figure: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
