import argparse
import sys
from agent import run_pipeline
from llm_advisor import generate_suggestion
from evaluate import evaluate
from output import replay
from config import DEFAULT_REPLAY_SPEED

def main():
    parser = argparse.ArgumentParser(description="Smart Factory Alert Agent CLI")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode (ignores --no-llm and --replay-speed)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM suggestions and use templates")
    parser.add_argument("--replay-speed", type=float, default=DEFAULT_REPLAY_SPEED,
                        help=f"Interval between alerts in seconds (default: {DEFAULT_REPLAY_SPEED})")

    args = parser.parse_args()

    # 1. Run the pipeline to get data
    results = run_pipeline()
    alerts_df = results["alerts_df"]
    test_df = results["test_df"]

    # 2. Handle mutually exclusive modes
    if args.evaluate:
        # Evaluation mode takes priority
        print("--- Running Evaluation Mode ---")
        metrics = evaluate(test_df)

        print("\nDetector Performance Metrics:")
        for detector, m in metrics.items():
            print(f"[{detector.upper()}] Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f}")

        print(f"\nConfusion matrices saved to docs/confusion_matrices.png")
        return

    # 3. Alerting Mode (Default or --no-llm)
    if alerts_df.empty:
        print("No anomalies detected. System is operating normally.")
        return

    use_llm = not args.no_llm
    suggestions = []

    # Process each alert to get a suggestion
    for _, row in alerts_df.iterrows():
        # record is passed as a dict
        record = row.to_dict()
        suggestion_text, is_fallback = generate_suggestion(record, use_llm=use_llm)
        suggestions.append((suggestion_text, is_fallback))

    # Replay the alerts to CLI and log
    replay(alerts_df, suggestions, replay_speed=args.replay_speed)

if __name__ == "__main__":
    main()
