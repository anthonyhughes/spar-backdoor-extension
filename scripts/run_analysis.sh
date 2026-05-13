#!/bin/bash
python scripts/collect_eval_results.py --csv tmp/eval_results.csv --best
python scripts/plot_eval_results.py
