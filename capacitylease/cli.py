from __future__ import annotations

import argparse

from .paper_audit import paper_audit
from .runner import diagnostics, reproduce


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CapacityLease simulation and verification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce_parser = subparsers.add_parser("reproduce", help="Generate plots and raw outputs")
    reproduce_parser.add_argument("--config", required=True, help="Path to JSON config file")
    reproduce_parser.add_argument("--project-root", required=True, help="Repository root")
    reproduce_parser.add_argument("--n-jobs", type=int, default=1, help="Parallel worker count")

    diagnostics_parser = subparsers.add_parser("diagnostics", help="Generate diagnostics and stability reports")
    diagnostics_parser.add_argument("--config", required=True, help="Path to JSON config file")
    diagnostics_parser.add_argument("--project-root", required=True, help="Repository root")
    diagnostics_parser.add_argument("--n-jobs", type=int, default=1, help="Parallel worker count")

    paper_audit_parser = subparsers.add_parser("paper-audit", help="Compare strict-table and narrative-matching configs")
    paper_audit_parser.add_argument("--project-root", required=True, help="Repository root")
    paper_audit_parser.add_argument("--n-jobs", type=int, default=1, help="Parallel worker count")
    paper_audit_parser.add_argument("--configs", nargs="*", default=None, help="Optional config paths")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "reproduce":
        reproduce(config_path=args.config, project_root=args.project_root, n_jobs=args.n_jobs)
    elif args.command == "diagnostics":
        diagnostics(config_path=args.config, project_root=args.project_root, n_jobs=args.n_jobs)
    elif args.command == "paper-audit":
        paper_audit(project_root=args.project_root, n_jobs=args.n_jobs, config_paths=args.configs)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
