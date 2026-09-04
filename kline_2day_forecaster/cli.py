"""Command line entry point: ``python -m kline_2day_forecaster.cli ...``"""

from __future__ import annotations

import argparse
import json

from .config import ForecastConfig
from .pipeline import train_forecaster


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-5m-bar two-day upside/downside regressors")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/kline_2day_forecaster")
    parser.add_argument("--symbol", default="TQQQ")
    parser.add_argument("--train-start-date")
    parser.add_argument("--train-end-date")
    parser.add_argument("--test-start-date")
    parser.add_argument("--test-end-date")
    parser.add_argument("--no-chan", action="store_true", help="Disable Chan enrichment for a fast baseline")
    parser.add_argument("--save-enriched-csv", action="store_true")
    args = parser.parse_args()
    cfg = ForecastConfig(input_csv=args.input_csv, output_dir=args.output_dir, symbol=args.symbol,
                         enable_chan=not args.no_chan, save_enriched_csv=args.save_enriched_csv,
                         train_start_date=args.train_start_date, train_end_date=args.train_end_date,
                         test_start_date=args.test_start_date, test_end_date=args.test_end_date)
    print(json.dumps(train_forecaster(cfg), indent=2))


if __name__ == "__main__":
    main()
