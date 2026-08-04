#!/usr/bin/env python3
"""
Entry point for the Metadata Generation Pipeline (Pipeline 0).

Executes the runner which handles argument parsing, configuration loading,
and orchestration.
"""

import sys
from metadata_gen.runner import main

if __name__ == "__main__":
    sys.exit(main())
