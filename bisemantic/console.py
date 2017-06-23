"""
Command line interface
"""

import argparse
import os
import textwrap

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
    parser.add_argument("--log", metavar="LEVEL", default="WARNING", help="logging level")
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Text Pair Classifier")

    # Shared arguments.
    data_arguments = argparse.ArgumentParser(add_help=False)
    data_group = data_arguments.add_argument_group("input data file parsing options")
    data_group.add_argument("--text-1-name", metavar="NAME", help="column containing the first text pair element")
    data_group.add_argument("--text-2-name", metavar="NAME", help="column containing the second text pair element")
    data_group.add_argument("--label-name", metavar="NAME", help="column containing the label")
    data_group.add_argument("--index-name", metavar="NAME",
                            help="column containing a unique index (default use row number)")
    data_group.add_argument("--invalid-labels", metavar="LABEL", nargs="*",
                            help="omit samples with these label values")
    data_group.add_argument("--not-comma-delimited", action="store_true", help="the data is not comma delimited")

    embedding_arguments = argparse.ArgumentParser(add_help=False)
    embedding_arguments.add_argument("--batch-size", metavar="SIZE", type=int, default=2048,
                                     help="number samples per batch (default 2048)")

    training_arguments = argparse.ArgumentParser(add_help=False)
    training_arguments.add_argument("training", metavar="TRAINING", help="training data file")
    training_group = training_arguments.add_argument_group("training options")
    training_group.add_argument("--epochs", type=int, default=10, help="training epochs (default 10)")
    training_group.add_argument("--n", type=int, help="number of training samples to use (default all)")
    validation_group = training_group.add_mutually_exclusive_group()
    validation_group.add_argument("--validation-set", metavar="FILE",
                                  help="validation data file (default no validation)")
    validation_group.add_argument("--validation-fraction", metavar="FRACTION", type=float,
                                  help="portion of the training data to use as validation (default no validation)")

    # Train subcommand
    train_parser = subparsers.add_parser("train", description=textwrap.dedent("""\
    Train a model to classify pairs of text.
    
    Training data is in a CSV document with column labels text1, text2, and label.
    Command line options may be used to specify different column labels.
    
    The generated model is saved in a directory.
    
    You may optionally specify either a separate labeled data file for validation or a portion of the training data
    to use as validation."""), parents=[data_arguments, training_arguments, embedding_arguments], help="train a model")
    train_parser.add_argument("--model-directory-name", metavar="DIRECTORY",
                              help="output model directory (default do not save a model)")
    model_group = train_parser.add_argument_group("model configuration options")
    model_group.add_argument("--units", type=int, default=128, help="LSTM hidden layer size (default 128)")
    model_group.add_argument("--dropout", type=float, help="Dropout rate (default no dropout)")
    model_group.add_argument("--maximum-tokens", metavar="TOKENS", type=int,
                             help="maximum number of tokens to embed per sample (default longest in the data)")
    model_group.add_argument("--bidirectional", action="store_true",
                             help="make LSTM bidirectional (default not bidirectional)")
    train_parser.set_defaults(func=lambda args: train(args))

    # Continue subcommand
    continue_parser = subparsers.add_parser("continue", description=textwrap.dedent("""\
    Continue training a model.
    
    The updated model information is written to the original model directory."""),
                                            parents=[data_arguments, training_arguments, embedding_arguments],
                                            help="continue training a model")
    continue_parser.add_argument("model_directory_name", metavar="MODEL",
                                 help="directory containing previously trained model")
    continue_parser.set_defaults(func=lambda args: continue_training(args))

    test_arguments = argparse.ArgumentParser(add_help=False)
    test_arguments.add_argument("model_directory_name", metavar="MODEL", help="model directory")
    test_arguments.add_argument("test", metavar="TEST", help="test data")
    test_arguments.add_argument("--n", type=int, help="number of test samples to use (default all)")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", description=textwrap.dedent("""\
    Use a model to predict a probability distribution over the text pair labels."""),
                                           parents=[data_arguments, embedding_arguments, test_arguments],
                                           help="predict labels")
    predict_parser.set_defaults(func=lambda args: predict(args))

    # Score subcommand
    predict_parser = subparsers.add_parser("score", description=textwrap.dedent("""\
    Use a model to score a labeled test set.
    
    This returns the model's cross entropy loss and accuracy on the test set."""),
                                           parents=[data_arguments, embedding_arguments, test_arguments],
                                           help="score labeled test set")
    predict_parser.set_defaults(func=lambda args: score(args))

    # Cross-validation subcommand
    cv_parser = subparsers.add_parser("cross-validation", description=textwrap.dedent("""\
    Create cross validation data partitions.
    
    These are written to CSV files in the specified output director."""), parents=[data_arguments],
                                      help="create cross validation")
    cv_parser.add_argument("data", metavar="DATA", help="data to partition")
    cv_parser.add_argument("fraction", metavar="FRACTION", type=float, help="fraction of the data to use for training")
    cv_parser.add_argument("k", metavar="K", type=int, help="number of splits")
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
                      TextPairClassifier.train(training, args.bidirectional, args.units, args.epochs,
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

    training = data_file(args.training, args.n, args.index_name, args.text_1_name, args.text_2_name, args.label_name,
                         args.invalid_labels, not args.not_comma_delimited)
    if args.validation_fraction is not None:
        training, validation = cross_validation_partitions(training, 1 - args.validation_fraction, 1)[0]
    elif args.validation_set is not None:
        validation = data_file(args.validation_set, args.n, args.index_name,
                               args.text_1_name, args.text_2_name, args.label_name, args.invalid_labels,
                               not args.not_comma_delimited)
    else:
        validation = None

    _, training_history = training_operation(args, training, validation)
    print(training_history.latest_run_summary())


def predict(args):
    from bisemantic.classifier import TextPairClassifier

    test = data_file(args.test, args.n, args.index_name, args.text_1_name, args.text_2_name, args.label_name,
                     args.invalid_labels, not args.not_comma_delimited)
    logger.info("Predict labels for %d pairs" % len(test))
    model = TextPairClassifier.load_from_model_directory(args.model_directory_name)
    class_names = TextPairClassifier.class_names_from_model_directory(args.model_directory_name)
    predictions = model.predict(test, batch_size=args.batch_size, class_names=class_names)
    print(predictions.to_csv())


def score(args):
    from bisemantic.classifier import TextPairClassifier

    test = data_file(args.test, args.n, args.index_name, args.text_1_name, args.text_2_name, args.label_name,
                     args.invalid_labels, not args.not_comma_delimited)
    logger.info("Score predictions for %d pairs" % len(test))
    model = TextPairClassifier.load_from_model_directory(args.model_directory_name)
    scores = model.score(test, batch_size=args.batch_size)
    print(", ".join("%s=%0.5f" % s for s in scores))


def create_cross_validation_partitions(args):
    from bisemantic.data import cross_validation_partitions
    data = data_file(args.data, args.n, args.index_name, args.text_1_name, args.text_2_name, args.label_name,
                     args.invalid_labels, not args.not_comma_delimited)
    for i, (train_partition, validate_partition) in enumerate(cross_validation_partitions(data, args.fraction, args.k)):
        train_name, validate_name = [os.path.join(args.output_directory, "%s.%d.%s.csv" % (args.prefix, i + 1, name))
                                     for name in ["train", "validate"]]
        train_partition.to_csv(train_name)
        validate_partition.to_csv(validate_name)
