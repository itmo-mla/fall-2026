import argparse
import os

from lin_clf.experiments.config import TrainConfig
from lin_clf.experiments.trainers.corr import train_corr
from lin_clf.experiments.trainers.fetch_prob import train_fetch_prob
from lin_clf.experiments.trainers.multistart import train_multistart
from lin_clf.experiments.runner import TrainResult
from lin_clf.experiments.trainers.speed_sgd import train_speed_sgd
from lin_clf.experiments.storage import save_result
from lin_clf.experiments.suite import run_suite, write_summary

_DEFAULTS = TrainConfig()

EXPERIMENTS = {
    "corr": train_corr,
    "speed-sgd": train_speed_sgd,
    "multistart": train_multistart,
    "fetch-prob": train_fetch_prob,
}


def _add_common_args(parser, out_default=None):
    parser.add_argument("--epochs", type=int, default=_DEFAULTS.epochs)
    parser.add_argument("--h", type=float, default=_DEFAULTS.h)
    parser.add_argument("--tao", type=float, default=_DEFAULTS.tao)
    parser.add_argument("--momentum-k", type=float, default=_DEFAULTS.momentum_k)
    parser.add_argument("--batch-size", type=int, default=_DEFAULTS.batch_size)
    parser.add_argument("--visual-smooth", type=float, default=_DEFAULTS.visual_smooth)
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument("--train-name", default=_DEFAULTS.train_name)
    parser.add_argument("--out", default=out_default)
    parser.add_argument("--max-points", type=int, default=1000)
    parser.add_argument("--no-draw", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(prog="train")
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    for name in EXPERIMENTS:
        subparser = subparsers.add_parser(name)
        _add_common_args(subparser)
        if name == "multistart":
            subparser.add_argument("--restarts", type=int, default=10)

    suite_parser = subparsers.add_parser("suite")
    _add_common_args(suite_parser, out_default="runs")
    suite_parser.add_argument("--restarts", type=int, default=10)
    return parser


def _build_config(args):
    return TrainConfig(
        epochs=args.epochs,
        h=args.h,
        tao=args.tao,
        momentum_k=args.momentum_k,
        batch_size=args.batch_size,
        visual_smooth=args.visual_smooth,
        seed=args.seed,
        draw_loss=not args.no_draw,
        train_name=args.train_name,
    )


def _run_single(name, config, args):
    result = EXPERIMENTS[name](config)
    print(
        f"{name}: epochs_run={result.epochs_run} "
        f"test_accuracy={result.test_accuracy:.4f}"
    )
    if args.out:
        print(f"saved {save_result(result, args.out, args.max_points, name)}")


def _run_multistart(config, args):
    result = train_multistart(config, restarts=args.restarts)
    print(
        f"multistart: restarts={args.restarts} "
        f"mean={result.mean:.4f} std={result.std:.4f}"
    )
    if args.out:
        for i, run_result in enumerate(result.results):
            run_name = f"restart_{i}"
            directory = os.path.join(args.out, run_name)
            print(
                f"saved {save_result(run_result, directory, args.max_points, run_name)}"
            )


def _run_suite(config, args):
    result = run_suite(config, restarts=args.restarts)
    for name, run_result in result.runs.items():
        save_result(run_result, os.path.join(args.out, name), args.max_points, name)
        print(f"{name}: test_accuracy={run_result.test_accuracy:.4f}")
    summary = write_summary(result, args.out)
    print(
        f"best: {result.best_name} "
        f"test_accuracy={result.runs[result.best_name].test_accuracy:.4f}"
    )
    print(f"saved {summary}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = _build_config(args)
    if args.experiment == "multistart":
        _run_multistart(config, args)
    elif args.experiment == "suite":
        _run_suite(config, args)
    else:
        _run_single(args.experiment, config, args)


if __name__ == "__main__":
    main()
