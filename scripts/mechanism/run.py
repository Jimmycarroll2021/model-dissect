#!/usr/bin/env python3
"""Guarded MECHANISM-001 CLI. No automatic admission or budget reset.

PYTHONPATH=src python scripts/mechanism/run.py observe --protocol RUN.json
  --inputs DEV.json --output NEW.json
PYTHONPATH=src python scripts/mechanism/run.py evaluate --protocol RUN.json
  --inputs INPUTS.json --hypothesis FROZEN.json --output NEW.json

Invoke through runtime/guard.py with exact reserved_input_positions and the
preapproved slot timeout. Identity validation precedes checkpoint loading.
"""
import argparse
from minimind_re.mechanism import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('observe', 'evaluate'))
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hypothesis')
    args = parser.parse_args()
    run(args.stage, args.protocol, args.inputs, args.output, args.hypothesis)


if __name__ == '__main__':
    main()
