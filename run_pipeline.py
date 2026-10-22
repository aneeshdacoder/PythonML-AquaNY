"""Run the whole pipeline (or selected stages) end to end.

  python run_pipeline.py --state NY --start 2015-01-01
  python run_pipeline.py --stages features train evaluate predict
"""
import argparse
import logging

import build_features
import collect_data
import config
import evaluate
import predict
import train

STAGES = ["collect", "features", "train", "evaluate", "predict"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--state", default="NY")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-sites", type=int, default=50)
    ap.add_argument("--min-params", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=config.HORIZON_DAYS)
    a = ap.parse_args()

    if "collect" in a.stages:
        collect_data.run(a.state, a.start, a.end, a.max_sites, a.min_params)
    if "features" in a.stages:
        build_features.run(a.horizon)
    if "train" in a.stages:
        train.run()
    if "evaluate" in a.stages:
        evaluate.run()
    if "predict" in a.stages:
        predict.run()


if __name__ == "__main__":
    main()
