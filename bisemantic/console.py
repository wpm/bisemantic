import argparse
import json
import logging
import os
import textwrap

import pandas as pd

import bisemantic
from bisemantic import load_data, TextualEquivalenceModel


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
    train_parser.add_argument("--units", type=int, default=128, help="LSTM hidden layer size (default 128)")
    train_parser.add_argument("--epochs", type=int, default=10, help="training epochs (default 10)")
    train_parser.add_argument("--model_directory_name", metavar="model", type=output_directory,
                              help="output model directory")
    train_parser.add_argument("--n", type=int, help="number of training samples to use (default all)")
    train_parser.set_defaults(func=lambda args: train(args))

    predict_parser = subparsers.add_parser("predict", description=textwrap.dedent("""\
    Use a model to predict textual equivalence."""), help="predict equivalence")
    predict_parser.add_argument("model", type=model_directory, help="model directory")
    predict_parser.add_argument("test", type=load_data, help="test data")
    predict_parser.add_argument("--n", type=int, help="number of test samples to use (default all)")
    predict_parser.set_defaults(func=lambda args: predict(args))

    return parser


def train(args):
    training = args.training.head(args.n)
    logging.info("Train on %d samples" % len(training))
    model, history = TextualEquivalenceModel.train(training, args.units, args.epochs, args.validation)
    history = history.history
    if args.model_directory_name is not None:
        logging.info("Save model to %s" % args.model_directory_name)
        model.save(model_filename(args.model_directory_name))
        with open(history_filename(args.model_directory_name), mode="w") as f:
            json.dump(history, f, sort_keys=True, indent=4, separators=(',', ': '))
    print("Training\naccuracy=%0.4f, loss=%0.4f" % (history["acc"][-1], history["loss"][-1]))
    if args.validation is not None:
        print("Validation\naccuracy=%0.4f, loss=%0.4f" % (history["val_acc"][-1], history["val_loss"][-1]))


def predict(args):
    test = args.test.head(args.n)
    logging.info("Predict labels for %d pairs" % len(test))
    model, _ = args.model
    print(pd.DataFrame({"predicted": model.predict(test)}).to_csv())


def output_directory(directory_name):
    os.makedirs(directory_name)
    return directory_name


def model_directory(directory_name):
    model = TextualEquivalenceModel.load(model_filename(directory_name))
    with open(history_filename(directory_name)) as f:
        history = json.load(f)
    return model, history


def model_filename(directory_name):
    return os.path.join(directory_name, "model.h5")


def history_filename(directory_name):
    return os.path.join(directory_name, "history.json")
