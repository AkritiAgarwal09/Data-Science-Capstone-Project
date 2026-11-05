"""Run the agent pipeline through the verified evaluation report."""

import argparse

from .graph import build_graph, config_for_dataset, run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument(
        "--diagram", action="store_true", help="print the LangGraph Mermaid diagram"
    )
    args = parser.parse_args()
    if args.diagram:
        print(build_graph().get_graph().draw_mermaid())
        return 0
    if not args.dataset:
        parser.error("--dataset is required unless --diagram is used")
    state, run = run_pipeline(config_for_dataset(args.dataset))
    print(f"{'FAILED' if state.failed_stage else 'SUCCESS'}: {run.root}")
    return 1 if state.failed_stage else 0


if __name__ == "__main__":
    raise SystemExit(main())
