import os
import shutil
import sys
import tempfile
from argparse import Namespace
from io import StringIO
from itertools import islice
from unittest import TestCase

import pandas as pd
from keras.callbacks import History
from numpy.testing import assert_array_equal

from bisemantic.console import data_file, main, fix_columns, TrainingHistory
from bisemantic.data import cross_validation_partitions, TextPairEmbeddingGenerator
from bisemantic.main import TextualEquivalenceModel


class TestPreprocess(TestCase):
    def setUp(self):
        self.train = data_file("test/resources/train.csv")

    def test_load_training_data(self):
        self.assertIsInstance(self.train, pd.DataFrame)
        assert_array_equal(["text1", "text2", "label"], self.train.columns)
        self.assertEqual(100, len(self.train))

    def test_load_test_data(self):
        actual = data_file("test/resources/test.csv")
        self.assertIsInstance(actual, pd.DataFrame)
        assert_array_equal(["text1", "text2"], actual.columns)
        self.assertEqual(9, len(actual))

    def test_load_data_with_null(self):
        actual = data_file("test/resources/data_with_null_values.csv")
        self.assertIsInstance(actual, pd.DataFrame)
        assert_array_equal(["text1", "text2", "label"], actual.columns)
        self.assertEqual(3, len(actual))

    def test_fix_columns_with_no_rename(self):
        train = fix_columns(self.train, Namespace(text_1_name=None, text_2_name=None, label_name=None))
        assert_array_equal(["text1", "text2", "label"], train.columns)

    def test_fix_columns_with_invalid_column_name(self):
        self.assertRaises(ValueError, fix_columns, self.train,
                          Namespace(text_1_name="bogus", text_2_name=None, label_name=None))

    def test_cross_validate(self):
        k = 3
        splits = cross_validation_partitions(self.train, 0.8, k)
        self.assertEqual(k, len(splits))
        for i in range(k):
            s = splits[i]
            self.assertIsInstance(s[0], pd.DataFrame)
            self.assertIsInstance(s[1], pd.DataFrame)
            self.assertEqual(80, len(s[0]))
            self.assertEqual(20, len(s[1]))


class TestTextPairEmbeddingGenerator(TestCase):
    def setUp(self):
        self.labeled = data_file("test/resources/train.csv")
        self.unlabeled = data_file("test/resources/test.csv")

    def test_embed_unlabeled(self):
        g = TextPairEmbeddingGenerator(self.unlabeled, batch_size=4)
        self.assertEqual(9, len(g))
        self.assertEqual("TextPairEmbeddingGenerator: 9 samples, batch size 4, maximum tokens 20", str(g))
        self.assertEqual(3, g.batches_per_epoch)
        two_epochs = list(islice(g(), 2 * g.batches_per_epoch))
        self._validate_unlabeled_batches(two_epochs, g.batches_per_epoch, 20, [4, 4, 1] * 2)

    def test_embed_labeled(self):
        g = TextPairEmbeddingGenerator(self.labeled, batch_size=32)
        self.assertEqual(100, len(g))
        self.assertEqual("TextPairEmbeddingGenerator: 100 samples, 2 classes, batch size 32, maximum tokens 40", str(g))
        self.assertEqual(4, g.batches_per_epoch)
        two_epochs = list(islice(g(), 2 * g.batches_per_epoch))
        self._validate_labeled_batches(two_epochs, g.batches_per_epoch, 40, [32, 32, 32, 4] * 2)

    def test_embed_labeled_specified_maximum_tokens(self):
        g = TextPairEmbeddingGenerator(self.labeled, batch_size=32, maximum_tokens=10)
        self.assertEqual(100, len(g))
        self.assertEqual("TextPairEmbeddingGenerator: 100 samples, 2 classes, batch size 32, maximum tokens 10", str(g))
        self.assertEqual(4, g.batches_per_epoch)
        two_epochs = list(islice(g(), 2 * g.batches_per_epoch))
        self._validate_labeled_batches(two_epochs, g.batches_per_epoch, 10, [32, 32, 32, 4] * 2)

    def _validate_unlabeled_batches(self, batches, batches_per_epoch, expected_maximum_tokens,
                                    expected_batch_sizes):
        # Verify that we got the expected data.
        for i, batch in enumerate(batches):
            self.assertIsInstance(batch, list)
            self.assertEqual(2, len(batch))
            expected_batch_size = expected_batch_sizes[i]
            self.assertEqual((expected_batch_size, expected_maximum_tokens, 300), batch[0].shape)
            self.assertEqual((expected_batch_size, expected_maximum_tokens, 300), batch[1].shape)
        # Verify that the data repeats itself after one epoch.
        for i, j in ((k, k + batches_per_epoch) for k in range(batches_per_epoch) if
                     k + batches_per_epoch < len(batches)):
            embedding_1_a = batches[i][0]
            embedding_1_b = batches[j][0]
            embedding_2_a = batches[i][1]
            embedding_2_b = batches[j][1]
            assert_array_equal(embedding_1_a, embedding_1_b)
            assert_array_equal(embedding_2_a, embedding_2_b)

    def _validate_labeled_batches(self, batches, batches_per_epoch, expected_maximum_tokens,
                                  expected_batch_sizes):
        # Verify that we got the expected data.
        for i, batch in enumerate(batches):
            self.assertIsInstance(batch, tuple)
            self.assertEqual(2, len(batch))
            self.assertIsInstance(batch[0], list)
            self.assertEqual(2, len(batch[0]))
            expected_batch_size = expected_batch_sizes[i]
            self.assertEqual((expected_batch_size, expected_maximum_tokens, 300), batch[0][0].shape)
            self.assertEqual((expected_batch_size, expected_maximum_tokens, 300), batch[0][1].shape)
            self.assertEqual((expected_batch_size,), batch[1].shape)
        # Verify that the data repeats itself after one epoch.
        for i, j in ((k, k + batches_per_epoch) for k in range(batches_per_epoch) if
                     k + batches_per_epoch < len(batches)):
            embedding_1_a = batches[i][0][0]
            embedding_1_b = batches[j][0][0]
            embedding_2_a = batches[i][0][1]
            embedding_2_b = batches[j][0][1]
            label_a = batches[i][1]
            label_b = batches[j][1]
            assert_array_equal(embedding_1_a, embedding_1_b)
            assert_array_equal(embedding_2_a, embedding_2_b)
            assert_array_equal(label_a, label_b)


class TestModel(TestCase):
    def setUp(self):
        data = data_file("test/resources/train.csv")
        n = int(0.8 * len(data))
        self.train = data[:n]
        self.validate = data[n:]
        self.test = data_file("test/resources/test.csv")
        self.temporary_directory = tempfile.mkdtemp()
        self.model_directory = os.path.join(self.temporary_directory, "model")

    def test_properties(self):
        model = TextualEquivalenceModel.create(2, 40, 300, 128, 0.5)
        self.assertEqual(40, model.maximum_tokens)
        self.assertEqual(300, model.embedding_size)
        self.assertEqual(128, model.lstm_units)
        self.assertEqual(0.5, model.dropout)
        self.assertEqual(2, model.classes)

    def test_stringification(self):
        model = TextualEquivalenceModel.create(2, 40, 300, 128, 0.5)
        self.assertEqual(
            "TextualEquivalenceModel(classes = 2, LSTM units = 128, maximum tokens = 40, embedding size = 300, dropout = 0.50)",
            str(model))
        model = TextualEquivalenceModel.create(2, 40, 300, 128, None)
        self.assertEqual(
            "TextualEquivalenceModel(classes = 2, LSTM units = 128, maximum tokens = 40, embedding size = 300, No dropout)",
            str(model))

    # noinspection PyUnresolvedReferences
    def test_train_and_predict(self):
        model, history = TextualEquivalenceModel.train(self.train, 128, 2,
                                                       dropout=0.5, maximum_tokens=30,
                                                       validation_data=self.validate,
                                                       model_directory=self.model_directory)
        self.assertEqual(
            "TextualEquivalenceModel(classes = 2, LSTM units = 128, maximum tokens = 30, embedding size = 300, dropout = 0.50)",
            str(model))
        self.assertIsInstance(model, TextualEquivalenceModel)
        self.assertIsInstance(history, History)
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.info.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        predictions = model.predict(self.test)
        self.assertEqual((len(self.test), 2), predictions.shape)
        self.assertTrue((predictions >= 0).all())
        self.assertTrue((predictions <= 1).all())

    def test_train_no_validation(self):
        model, history = TextualEquivalenceModel.train(self.train.head(20), 128, 1, dropout=0.5,
                                                       maximum_tokens=30, model_directory=self.model_directory)
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.info.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        self.assertIsInstance(model, TextualEquivalenceModel)
        self.assertIsInstance(history, History)

    def test_train_no_model_directory(self):
        model, history = TextualEquivalenceModel.train(self.train.head(20), 128, 1, dropout=0.5, maximum_tokens=30)
        self.assertIsInstance(model, TextualEquivalenceModel)
        self.assertIsInstance(history, History)

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


class TestSerialization(TestCase):
    def setUp(self):
        _, self.filename = tempfile.mkstemp('.h5')

    def test_serialization(self):
        model = TextualEquivalenceModel.create(2, 40, 300, 128, 0.5)
        model.save(self.filename)
        deserialized_model = TextualEquivalenceModel.load(self.filename)
        self.assertIsInstance(deserialized_model, TextualEquivalenceModel)
        self.assertEqual(model.maximum_tokens, deserialized_model.maximum_tokens)
        self.assertEqual(model.embedding_size, deserialized_model.embedding_size)
        self.assertEqual(model.lstm_units, deserialized_model.lstm_units)

    def tearDown(self):
        os.remove(self.filename)


class TestCommandLine(TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.model_directory = os.path.join(self.temporary_directory, "model")

    def test_no_arguments(self):
        actual = main_function_output([])
        self.assertEqual(
            "usage: bisemantic [-h] [--version] [--log LOG]\n                  " +
            "{train,continue,predict,cross-validation} ...\n", actual)

    def test_version(self):
        actual = main_function_output(["--version"])
        self.assertEqual("""bisemantic 1.0.0\n""", actual)

    def test_cross_validation(self):
        main_function_output(["cross-validation", "test/resources/train.csv",
                              "0.8", "3",
                              "--prefix", "_batches",
                              "--output-directory", self.temporary_directory])
        for i in range(1, 3):
            for partition_name in ["train", "validate"]:
                filename = os.path.join(self.temporary_directory, "_batches.%d.%s.csv" % (i, partition_name))
                self.assertTrue(os.path.isfile(filename), "%s is not a file" % filename)

    def test_train_predict(self):
        main_function_output(["train", "test/resources/train.csv",
                              "--validation-set", "test/resources/train.csv",
                              "--units", "64",
                              "--dropout", "0.5",
                              "--epochs", "2",
                              "--model", self.model_directory])
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        training_history_filename = os.path.join(self.model_directory, "training-history.json")
        self.assertTrue(os.path.isfile(training_history_filename))
        training_history = TrainingHistory.load(training_history_filename)
        self.assertEqual("Training history, 1 runs", str(training_history))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.info.txt")))
        main_function_output(["predict", self.model_directory, "test/resources/test.csv"])

    def test_train_predict_no_validation(self):
        main_function_output(["train", "test/resources/train.csv",
                              "--units", "64",
                              "--dropout", "0.5",
                              "--epochs", "2",
                              "--model", self.model_directory])
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        training_history_filename = os.path.join(self.model_directory, "training-history.json")
        self.assertTrue(os.path.isfile(training_history_filename))
        training_history = TrainingHistory.load(training_history_filename)
        self.assertEqual("Training history, 1 runs", str(training_history))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.info.txt")))
        main_function_output(["predict", self.model_directory, "test/resources/test.csv"])

    def test_train_predict_crossvalidation_fraction_with_continue(self):
        # Train a model.
        main_function_output(["train", "test/resources/train.csv",
                              "--validation-fraction", "0.2",
                              "--units", "64",
                              "--dropout", "0.5",
                              "--epochs", "2",
                              "--model", self.model_directory])
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        training_history_filename = os.path.join(self.model_directory, "training-history.json")
        self.assertTrue(os.path.isfile(training_history_filename))
        training_history = TrainingHistory.load(training_history_filename)
        self.assertEqual("Training history, 1 runs", str(training_history))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.info.txt")))
        # Use it to make predictions on a test set.
        main_function_output(["predict", self.model_directory, "test/resources/test.csv"])
        # Train the model some more.
        main_function_output(["continue",
                              "test/resources/train.csv",
                              self.model_directory,
                              "--validation-fraction", "0.2",
                              "--epochs", "2"])
        training_history = TrainingHistory.load(training_history_filename)
        self.assertEqual("Training history, 2 runs", str(training_history))

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


def main_function_output(args):
    sys.argv = ["bisemantic"] + args
    sys.stdout = s = StringIO()
    try:
        main()
    except SystemExit:
        pass
    sys.stdout = sys.__stdout__
    return s.getvalue()
