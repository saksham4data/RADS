import argparse
import json
import logging

from rads.live.engine import RuntimeEngine


def _console_handler(event):
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description="RADS Deployable Runtime v1")
    parser.add_argument("--config", required=True, help="Path to pipeline configuration YAML")
    parser.add_argument("--source", type=str, default=None, help="Override source URI")
    parser.add_argument("--api", action="store_true", help="Enable API server")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = RuntimeEngine(args.config, source_uri=args.source, api_enabled=True if args.api else None)
    engine.register_handler(_console_handler)
    engine.run()


if __name__ == "__main__":
    main()
