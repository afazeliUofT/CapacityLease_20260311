from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_json, save_json
from .models import ModelSpec
from .runner import diagnostics, reproduce


def _relative_error_score(claim_check: dict[str, Any]) -> float:
    total = 0.0
    count = 0
    for key, target in claim_check.get("claims", {}).items():
        actual = claim_check.get("actual", {}).get(key)
        if target in (None, 0) or actual is None:
            continue
        total += abs(float(actual) - float(target)) / max(abs(float(target)), 1e-12)
        count += 1
    return total / max(count, 1)


def _load_or_build(config_path: Path, project_root: Path, n_jobs: int) -> dict[str, Any]:
    payload = load_json(config_path)
    spec = ModelSpec(payload)
    report_dir = project_root / "outputs" / spec.name / "reports"
    if not (report_dir / "summary.json").exists():
        reproduce(config_path=config_path, project_root=project_root, n_jobs=n_jobs)
    if not (report_dir / "solver_certification.json").exists():
        diagnostics(config_path=config_path, project_root=project_root, n_jobs=n_jobs)

    return {
        "config_name": spec.name,
        "config_path": str(config_path),
        "summary": load_json(report_dir / "summary.json"),
        "claim_check": load_json(report_dir / "claim_check.json"),
        "solver_certification": load_json(report_dir / "solver_certification.json"),
        "paper_table_alignment": load_json(report_dir / "paper_table_alignment.json"),
    }


def paper_audit(project_root: str | Path, n_jobs: int, config_paths: list[str] | None = None) -> dict[str, Any]:
    project_root = Path(project_root)
    if config_paths is None:
        config_paths = [
            str(project_root / "configs" / "paper_figures_consistent.json"),
            str(project_root / "configs" / "paper_strict_text.json"),
        ]

    results = [_load_or_build(Path(path), project_root, n_jobs=n_jobs) for path in config_paths]
    by_name = {entry["config_name"]: entry for entry in results}

    names = list(by_name)
    if len(names) < 2:
        raise ValueError("paper_audit requires at least two configs.")

    best_narrative_fit_name = min(names, key=lambda name: _relative_error_score(by_name[name]["claim_check"]))

    first = by_name[names[0]]
    second = by_name[names[1]]
    first_payload = load_json(first["config_path"])
    second_payload = load_json(second["config_path"])

    cross_config_parameter_differences = {
        "delta": float(first_payload["parameters"]["delta"]) - float(second_payload["parameters"]["delta"]),
        "C": float(first_payload["parameters"]["C"]) - float(second_payload["parameters"]["C"]),
        "zeta": float(first_payload["parameters"]["zeta"]) - float(second_payload["parameters"]["zeta"]),
        "lambda": float(first_payload["parameters"]["lambda"]) - float(second_payload["parameters"]["lambda"]),
        "pi_V": float(first_payload["parameters"]["pi_V"]) - float(second_payload["parameters"]["pi_V"]),
    }

    claim_fit = {
        name: {
            "mean_relative_claim_error": _relative_error_score(entry["claim_check"]),
            "claim_errors": entry["claim_check"].get("errors", {}),
        }
        for name, entry in by_name.items()
    }

    verdict = {
        "best_narrative_fit_config": best_narrative_fit_name,
        "single_config_matches_both_table_and_narrative": False,
        "table_exact_configs": [name for name, entry in by_name.items() if entry["paper_table_alignment"].get("matches_paper_simulation_table", False)],
        "configs_with_all_solver_certifications_true": [
            name
            for name, entry in by_name.items()
            if all(bool(value) for value in entry["solver_certification"].get("verdict", {}).values())
        ],
    }

    report = {
        "configs": by_name,
        "cross_config_parameter_differences": cross_config_parameter_differences,
        "claim_fit": claim_fit,
        "verdict": verdict,
    }

    out_dir = project_root / "outputs" / "paper_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "paper_config_comparison.json", report)
    return report
