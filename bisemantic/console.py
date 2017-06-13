"""
Command line interface
"""

import argparse
import json
import os
import textwrap
import time
from datetime import timedelta

import pandas as pd

import bisemantic
from bisemantic import text_1, text_2, label, configure_logger, logger


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

    column_renames = argparse.ArgumentParser(add_help=False)
    column_renames.add_argument("--text-1-name", metavar="NAME", help="column containing the first text pair element")
    column_renames.add_argument("--text-2-name", metavar="NAME", help="column containing the second text pair element")
    column_renames.add_argument("--label-name", metavar="NAME", help="column containing the label")

    text_parser_options = argparse.ArgumentParser(add_help=False)
    text_parser_options.add_argument("--parser-threads", metavar="THREADS", type=int, default=-1,
                                     help="number of parallel text parsing threads (default maximum possible)")
    text_parser_options.add_argument("--parser-batch-size", metavar="SAMPLES", type=int, default=1000,
                                     help="batch size passed to text parser")

    train_parser = subparsers.add_parser("train", description=textwrap.dedent("""\
    Train a model to predict textual equivalence."""), parents=[column_renames, text_parser_options],
                                         help="train model")
    train_parser.add_argument("training", type=data_file, help="training data")
    validation_group = train_parser.add_mutually_exclusive_group()
    validation_group.add_argument("--validation-set", type=data_file, help="validation data")
    validation_group.add_argument("--validation-fraction", type=float,
                                  help="portion of the training data to use as validation")
    train_parser.add_argument("--units", type=int, default=128, help="LSTM hidden layer size (default 128)")
    train_parser.add_argument("--dropout", type=float, help="Dropout rate (default no dropout)")
    train_parser.add_argument("--maximum-tokens", type=int, help="maximum number of tokens to embed per sample")
    train_parser.add_argument("--epochs", type=int, default=10, help="training epochs (default 10)")
    train_parser.add_argument("--model-directory-name", metavar="MODEL", type=output_directory,
                              help="output model directory")
    train_parser.add_argument("--gpu-fraction", type=float, help="apportion this fraction of GPU memory for a process")
    train_parser.add_argument("--n", type=int, help="number of training samples to use (default all)")
    train_parser.set_defaults(func=lambda args: train(args))

    predict_parser = subparsers.add_parser("predict", description=textwrap.dedent("""\
    Use a model to predict textual equivalence."""), parents=[column_renames, text_parser_options],
                                           help="predict equivalence")
    predict_parser.add_argument("model", type=model_directory, help="model directory")
    predict_parser.add_argument("test", type=data_file, help="test data")
    predict_parser.add_argument("--n", type=int, help="number of test samples to use (default all)")
    predict_parser.set_defaults(func=lambda args: predict(args))

    cv_parser = subparsers.add_parser("cross-validation", description=textwrap.dedent("""\
    Create cross validation data partitions."""), parents=[column_renames], help="cross validation")
    cv_parser.add_argument("data", type=data_file, help="data to partition")
    cv_parser.add_argument("fraction", type=float, help="fraction to use for training")
    cv_parser.add_argument("k", type=int, help="number of splits")
    cv_parser.add_argument("--prefix", type=str, default="data", help="name prefix of partition files (default data)")
    cv_parser.add_argument("--output-directory", metavar="DIRECTORY", type=str, default=".",
                           help="output directory (default working directory)")
    cv_parser.add_argument("--n", type=int, help="number of samples to use (default all)")
    cv_parser.set_defaults(func=lambda args: create_cross_validation_partitions(args))

    return parser


def train(args):
    if args.gpu_fraction is not None:
        _set_gpu_fraction(args)

    from bisemantic.main import cross_validation_partitions, TextualEquivalenceModel

    training = fix_columns(args.training.head(args.n), args)
    if args.validation_fraction is not None:
        training, validation = cross_validation_partitions(training, 1 - args.validation_fraction, 1)[0]
    else:
        validation = args.validation_set

    start = time.time()
    model, history = TextualEquivalenceModel.train(training, args.units, args.epochs,
                                                   args.dropout, args.maximum_tokens,
                                                   validation, args.model_directory_name,
                                                   args.parser_threads, args.parser_batch_size)
    training_time = str(timedelta(seconds=time.time() - start))
    history = history.history
    if args.model_directory_name is not None:
        logger.info("Save model to %s" % args.model_directory_name)
        with open(history_filename(args.model_directory_name), mode="w") as f:
            json.dump({"training-time": training_time,
                       "training-samples": len(training),
                       "model-parameters": model.parameters(),
                       "history": history}, f,
                      sort_keys=True, indent=4, separators=(',', ': '))
    print("Training time %s" % training_time)
    print("Training: accuracy=%0.4f, loss=%0.4f" % (history["acc"][-1], history["loss"][-1]))
    if validation is not None:
        print("Validation: accuracy=%0.4f, loss=%0.4f" % (history["val_acc"][-1], history["val_loss"][-1]))


def _set_gpu_fraction(args):
    """
    By default, TensorFlow allocates all the available GPU memory. If you want to run multiple processes on the same
    machine, you need to change it to allocate a fraction of the memory per process. This option only works when
    running on a GPU machine with the TensorFlow backend.
    """
    # noinspection PyPackageRequirements
    import tensorflow as tf
    from keras.backend import tensorflow_backend

    def get_session(gpu_fraction):
        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        if num_threads:
            return tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    tensorflow_backend.set_session(get_session(args.gpu_fraction))


def predict(args):
    test = fix_columns(args.test.head(args.n), args)
    logger.info("Predict labels for %d pairs" % len(test))
    model, _ = args.model
    predictions = model.predict(test, args.parser_threads, args.parser_batch_size)
    print(pd.DataFrame({"predicted": predictions}).to_csv())


def create_cross_validation_partitions(args):
    from bisemantic.main import cross_validation_partitions
    data = fix_columns(args.data.head(args.n), args)
    for i, (train_partition, validate_partition) in enumerate(cross_validation_partitions(data, args.fraction, args.k)):
        train_name, validate_name = [os.path.join(args.output_directory, "%s.%d.%s.csv" % (args.prefix, i + 1, name))
                                     for name in ["train", "validate"]]
        train_partition.to_csv(train_name)
        validate_partition.to_csv(validate_name)


def data_file(filename):
    """
    Load a test or training data file.

    A data file is a CSV file. Any rows with null values are dropped.

    :param filename: name of data file
    :type filename: str
    :return: data stored in the data file
    :rtype: pandas.DataFrame
    """
    data = pd.read_csv(filename)
    m = len(data)
    data = data.dropna()
    n = len(data)
    if m != n:
        logger.info("Dropped %d lines with null values from %s" % (m - n, filename))
    return data


# noinspection PyUnresolvedReferences
def fix_columns(data, args):
    """
    Rename columns in an input data frame to the ones bisemantic expects. Drop unused columns. If an argument is None
    the corresponding column must already be in the raw data.

    :param data: raw data
    :type data: pandas.DataFrame
    :param args: parsed command line arguments
    :type args: argparse.Namespace
    :return: data frame containing just the needed columns
    :rtype: pandas.DataFrame
    """
    for name in [args.text_1_name, args.text_2_name, args.label_name]:
        if name is not None:
            if name not in data.columns:
                raise ValueError("Missing column %s" % name)
    data = data.rename(columns={args.text_1_name: text_1, args.text_2_name: text_2, args.label_name: label})
    if label in data.columns:
        columns = [text_1, text_2, label]
    else:
        columns = [text_1, text_2]
    return data[columns]


def output_directory(directory_name):
    os.makedirs(directory_name)
    return directory_name


def model_directory(directory_name):
    from bisemantic.main import TextualEquivalenceModel
    model = TextualEquivalenceModel.load(model_filename(directory_name))
    with open(history_filename(directory_name)) as f:
        history = json.load(f)
    return model, history


def model_filename(directory_name):
    return os.path.join(directory_name, "model.h5")


def history_filename(directory_name):
    return os.path.join(directory_name, "history.json")
