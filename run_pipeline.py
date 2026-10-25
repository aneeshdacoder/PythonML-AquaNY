"""Run the pipeline end to end, or selected stages.

  python run_pipeline.py --source usgs --state NY --weather
  python run_pipeline.py --source csv --file export.csv --mapping examples/mapping_example.json
  python run_pipeline.py --stages features train evaluate predict monitor
"""
import argparse
import logging

import build_features
import config
import evaluate
import ingest
import monitor
import predict
import train

STAGES = ["ingest", "features", "train", "evaluate", "predict", "monitor"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--source", choices=["usgs", "csv"], default="usgs")
    ap.add_argument("--state", default="NY")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-sites", type=int, default=50)
    ap.add_argument("--min-params", type=int, default=1)
    ap.add_argument("--weather", action="store_true")
    ap.add_argument("--skip-lab", action="store_true")
    ap.add_argument("--file")
    ap.add_argument("--mapping")
    ap.add_argument("--stations-file")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--recent-days", type=int, default=30)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(config.HORIZONS))
    a = ap.parse_args()

    if "ingest" in a.stages:
        ingest.run(a.source, a.state, a.start, a.end, a.max_sites, a.min_params, not a.skip_lab,
                   a.weather, a.file, a.mapping, a.stations_file, a.append, a.recent_days)
    if "features" in a.stages:
        build_features.run(a.horizons)
    if "train" in a.stages:
        train.run(a.horizons)
    if "evaluate" in a.stages:
        evaluate.run()
    if "predict" in a.stages:
        predict.run(horizons=a.horizons)
    if "monitor" in a.stages:
        monitor.run()


if __name__ == "__main__":
    main()
