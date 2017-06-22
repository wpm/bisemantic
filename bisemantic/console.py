"""
Command line interface
"""

import argparse
import json
import os
import textwrap
import time
from datetime import timedelta, datetime

import pandas as pd

import bisemantic
from bisemantic import configure_logger, logger
from bisemantic.data import data_file


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    logger.info("Start")
    args.func(args)
    logger.info("Done")


def create_argument_parser():
    parser = argparse.ArgumentParser(description=bisemantic.__doc__)
    parser.add_argument('--version', action='version', version="%(prog)s " + bisemantic.__version__)
    parser.add_argument("--log", default="WARNING", help="logging level")
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Text Pair Equivalence")

    # Shared arguments.
    column_renames = argparse.ArgumentParser(add_help=False)
    column_renames.add_argument("--text-1-name", metavar="NAME", help="column containing the first text pair element")
    column_renames.add_argument("--text-2-name", metavar="NAME", help="column containing the second text pair element")
    column_renames.add_argument("--label-name", metavar="NAME", help="column containing the label")
    column_renames.add_argument("--index-name", metavar="NAME",
                                help="column containing a unique index (default use row number)")

    embedding_options = argparse.ArgumentParser(add_help=False)
    embedding_options.add_argument("--batch-size", type=int, default=2048,
                                   help="number samples per batch (default 2048)")

    model_parameters = argparse.ArgumentParser(add_help=False)
    model_parameters.add_argument("--units", type=int, default=128, help="LSTM hidden layer size (default 128)")
    model_parameters.add_argument("--dropout", type=float, help="Dropout rate (default no dropout)")
    model_parameters.add_argument("--maximum-tokens", type=int,
                                  help="maximum number of tokens to embed per sample (default longest in the data)")

    training_arguments = argparse.ArgumentParser(add_help=False)
    training_arguments.add_argument("training", help="training data")
    validation_group = training_arguments.add_mutually_exclusive_group()
    validation_group.add_argument("--validation-set", help="validation data")
    validation_group.add_argument("--validation-fraction", type=float,
                                  help="portion of the training data to use as validation")
    training_arguments.add_argument("--epochs", type=int, default=10, help="training epochs (default 10)")
    training_arguments.add_argument("--n", type=int, help="number of training samples to use (default all)")

    # Train subcommand
    train_parser = subparsers.add_parser("train", description=textwrap.dedent("""\
    Train a model to predict textual equivalence.
    
    Training data is in a CSV document with column labels text1, text2, and label.
    Command line options may be used to specify different column labels.
    
    The generated model is saved in a directory.
    
    You may optionally specify either a separate labeled data file for validation or a portion of the training data
    to use as validation."""),
                                         parents=[column_renames, model_parameters, training_arguments,
                                                  embedding_options],
                                         help="train model")
    train_parser.add_argument("--model-directory-name", metavar="MODEL", help="output model directory")
    train_parser.set_defaults(func=lambda args: train(args))

    # Continue subcommand
    continue_parser = subparsers.add_parser("continue", description=textwrap.dedent("""\
    Continue training a model.
    
    The updated model information is written to the original model directory."""),
                                            parents=[column_renames, training_arguments, embedding_options],
                                            help="continue training a model")
    continue_parser.add_argument("model_directory_name", metavar="MODEL",
                                 help="directory containing previously trained model")
    continue_parser.set_defaults(func=lambda args: continue_training(args))

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", description=textwrap.dedent("""\
    Use a model to predict the probability of the text pair label."""),
                                           parents=[column_renames, embedding_options],
                                           help="predict equivalence")
    predict_parser.add_argument("model_directory_name", metavar="MODEL", help="model directory")
    predict_parser.add_argument("test", help="test data")
    predict_parser.add_argument("--n", type=int, help="number of test samples to use (default all)")
    predict_parser.set_defaults(func=lambda args: predict(args))

    # Cross-validation subcommand
    cv_parser = subparsers.add_parser("cross-validation", description=textwrap.dedent("""\
    Create cross validation data partitions."""), parents=[column_renames], help="cross validation")
    cv_parser.add_argument("data", help="data to partition")
    cv_parser.add_argument("fraction", type=float, help="fraction to use for training")
    cv_parser.add_argument("k", type=int, help="number of splits")
    cv_parser.add_argument("--prefix", type=str, default="data", help="name prefix of partition files (default data)")
    cv_parser.add_argument("--output-directory", metavar="DIRECTORY", type=str, default=".",
                           help="output directory (default working directory)")
    cv_parser.add_argument("--n", type=int, help="number of samples to use (default all)")
    cv_parser.set_defaults(func=lambda args: create_cross_validation_partitions(args))

    return parser


def train(args):
    from bisemantic.classifier import TextPairClassifier
    train_or_continue(args,
                      lambda a, training, validation:
                      TextPairClassifier.train(training, args.units, args.epochs,
                                               dropout=args.dropout, maximum_tokens=args.maximum_tokens,
                                               batch_size=args.batch_size,
                                               validation_data=validation,
                                               model_directory=args.model_directory_name))


def continue_training(args):
    from bisemantic.classifier import TextPairClassifier
    train_or_continue(args,
                      lambda a, training, validation:
                      TextPairClassifier.continue_training(training, args.epochs, args.model_directory_name,
                                                           batch_size=args.batch_size, validation_data=validation))


def train_or_continue(args, training_operation):
    from bisemantic.data import cross_validation_partitions

    training = data_file(args.training, args.n, args.index_name, args.text_1_name, args.text_2_name, args.label_name)
    if args.validation_fraction is not None:
        training, validation = cross_validation_partitions(training, 1 - args.validation_fraction, 1)[0]
    elif args.validation_set is not None:
        validation = data_file(args.validation_set, args.n, args.index_name,
                               args.text_1_name, args.text_2_name, args.label_name)
    else:
        validation = None

    start = time.time()
    model, history = training_operation(args, training, validation)
    training_time = str(timedelta(seconds=time.time() - start))
    if args.model_directory_name is not None:
        update_model_directory(args.model_directory_name, training_time, len(training), history)
    print("Training time %s" % training_time)
    history = history.history
    if "val_loss" in history:
        metric = "val_loss"
    else:
        metric = "loss"
    i = history[metric].index(min(history[metric]))
    print("Best epoch: %d" % (i + 1))
    print("Training: accuracy=%0.4f, loss=%0.4f" % (history["acc"][i], history["loss"][i]))
    if validation is not None:
        print("Validation: accuracy=%0.4f, loss=%0.4f" % (history["val_acc"][i], history["val_loss"][i]))


def predict(args):
    from bisemantic.classifier import TextPairClassifier

    test = data_file(args.test, args.n, args.index_name, args.text_1_name, args.text_2_name, args.label_name)
    logger.info("Predict labels for %d pairs" % len(test))
    model = TextPairClassifier.load_from_model_directory(args.model_directory_name)
    predictions = model.predict(test, batch_size=args.batch_size)
    print(pd.DataFrame(predictions).to_csv())


def create_cross_validation_partitions(args):
    from bisemantic.data import cross_validation_partitions
    data = data_file(args.data, args.n, args.index_name, args.text_1_name, args.text_2_name, args.label_name)
    for i, (train_partition, validate_partition) in enumerate(cross_validation_partitions(data, args.fraction, args.k)):
        train_name, validate_name = [os.path.join(args.output_directory, "%s.%d.%s.csv" % (args.prefix, i + 1, name))
                                     for name in ["train", "validate"]]
        train_partition.to_csv(train_name)
        validate_partition.to_csv(validate_name)


def update_model_directory(directory_name, training_time, samples, history):
    training_history_filename = os.path.join(directory_name, "training-history.json")
    if os.path.isfile(training_history_filename):
        training_history = TrainingHistory.load(training_history_filename)
    else:
        training_history = TrainingHistory()
    training_history.add_run(training_time, samples, history)
    training_history.save(training_history_filename)


class TrainingHistory(object):
    """
    Record of all the training runs made on a given model. This records the training date, the size of the sample, and
    the training and validation scores.
    """

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            return cls(json.load(f))

    def __init__(self, runs=None):
        self.runs = runs or []

    def __repr__(self):
        return "Training history, %d runs" % (len(self.runs))

    def add_run(self, training_time, samples, history):
        self.runs.append({"training-time": training_time,
                          "samples": samples,
                          "history": history.history,
                          "run-date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.runs, f, sort_keys=True, indent=4, separators=(",", ": "))
