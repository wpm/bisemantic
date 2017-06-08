import argparse
import logging
import textwrap

import bisemantic
from bisemantic import load_data


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)-15s %(levelname)-8s %(message)s", level=args.log.upper())
    args.func(args)


def create_argument_parser():
    parser = argparse.ArgumentParser(description=bisemantic.__doc__)
    parser.add_argument('--version', action='version', version="%(prog)s " + bisemantic.__version__)
    parser.add_argument("--log", default="WARNING", help="logging level")
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Text Pair Equivalence")

    train_parser = subparsers.add_parser("train", description=textwrap.dedent("""\
    Train a model to predict textual equivalence."""), help="train model")
    train_parser.add_argument("training", type=load_data, help="training data")
    train_parser.add_argument("--validation", type=load_data, help="validation data")
    train_parser.add_argument("--lstm_units", metavar="lstm-units", type=int, default=128,
                              help="LSTM hidden layer size (default 128)")
    train_parser.add_argument("--model_directory_name", metavar="model", help="output model directory")
    train_parser.add_argument("--n", type=int, help="number of training samples to use (default all)")
    train_parser.set_defaults(func=lambda args: train(args))

    predict_parser = subparsers.add_parser("predict", description=textwrap.dedent("""\
    Use a model to predict textual equivalence."""), help="predict equivalence")
    predict_parser.add_argument("model_directory_name", metavar="model", help="model directory")
    predict_parser.add_argument("test", type=load_data, help="test data")
    predict_parser.add_argument("--n", type=int, help="number of test samples to use (default all)")
    predict_parser.set_defaults(func=lambda args: predict(args))

    return parser


def train(args):
    training = args.training.head(args.n)
    logging.debug("Train on %d samples" % len(training))
    validation = args.validation


def predict(args):
    test = args.test.head(args.n)
    logging.debug("Predict labels for %d pairs" % len(test))
