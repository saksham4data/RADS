import argparse
import json
import logging
import os

from rads.runtime.engine import RuntimeEngine


def _console_handler(event):
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description="RADS Deployable Runtime v1")
    parser.add_argument("--config", required=True, help="Path to pipeline configuration YAML")
    parser.add_argument("--source", type=str, default=None, help="Override source URI")
    parser.add_argument("--device", type=str, default=None, help="Override compute device")
    parser.add_argument("--api", action="store_true", help="Enable API server")
    args = parser.parse_args()
    if args.source is not None:
        os.environ["RADS_SOURCE_URI"] = args.source
    if args.device is not None:
        os.environ["RADS_DEVICE"] = args.device
    if args.api:
        os.environ["RADS_API_ENABLED"] = "true"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = RuntimeEngine(args.config)
    engine.register_handler(_console_handler)
    engine.run()


if __name__ == "__main__":
    main()
