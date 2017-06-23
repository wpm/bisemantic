import os
import shutil
import sys
import tempfile
from io import StringIO
from itertools import islice
from unittest import TestCase

import pandas as pd
from numpy import ones
from numpy.testing import assert_array_equal, assert_allclose

from bisemantic.classifier import TextPairClassifier, TrainingHistory
from bisemantic.console import main
from bisemantic.data import cross_validation_partitions, TextPairEmbeddingGenerator, data_file, load_data_file, \
    fix_columns


class TestPreprocess(TestCase):
    def setUp(self):
        self.train = load_data_file("test/resources/train.csv")

    def test_load_training_data(self):
        self.assertIsInstance(self.train, pd.DataFrame)
        assert_array_equal(["text1", "text2", "label"], self.train.columns)
        self.assertEqual(100, len(self.train))

    def test_load_test_data(self):
        actual = load_data_file("test/resources/test.csv")
        self.assertIsInstance(actual, pd.DataFrame)
        assert_array_equal(["text1", "text2"], actual.columns)
        self.assertEqual(9, len(actual))

    def test_load_data_with_null(self):
        actual = data_file("test/resources/data_with_null_values.csv")
        self.assertIsInstance(actual, pd.DataFrame)
        assert_array_equal(["text1", "text2", "label"], actual.columns)
        self.assertEqual(3, len(actual))

    def test_load_data_with_null_and_index_column(self):
        actual = data_file("test/resources/data_with_null_values.csv", index="id")
        self.assertIsInstance(actual, pd.DataFrame)
        assert_array_equal(["text1", "text2", "label"], actual.columns)
        self.assertEqual(3, len(actual))
        assert_array_equal([1, 3, 5], actual.index)

    def test_fix_columns_with_no_rename(self):
        train = fix_columns(self.train, text_1_name=None, text_2_name=None, label_name=None)
        assert_array_equal(["text1", "text2", "label"], train.columns)

    def test_fix_columns_with_invalid_column_name(self):
        self.assertRaises(ValueError, fix_columns, self.train, text_1_name="bogus", text_2_name=None, label_name=None)

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


class TestNonCommaDelimited(TestCase):
    # The Standford textual entailment SNLI format uses spaces as delimiters instead of commas.
    def test_load_data_with_space_delimiter(self):
        snli = load_data_file("test/resources/snli.csv", index="pairID", comma_delimited=False)
        self.assertEqual(10, len(snli))

    def test_load_snli_format(self):
        snli = data_file("test/resources/snli.csv", index="pairID",
                         text_1_name="sentence1", text_2_name="sentence2", label_name="gold_label",
                         invalid_labels=["-"], comma_delimited=False)
        assert_array_equal(["text1", "text2", "label"], snli.columns)
        # There are 9 samples because one is dropped because it has "-" as a gold_label.
        self.assertEqual(9, len(snli))
        # The pairID index is names of JPEGs.
        self.assertTrue(snli.index.str.match(r"\d{10}\.jpg#\dr\d[nec]").all())
        self.assertEqual({'contradiction', 'neutral', 'entailment'}, set(snli.label.values))


class TestTextPairEmbeddingGenerator(TestCase):
    def setUp(self):
        self.labeled = load_data_file("test/resources/train.csv")
        self.unlabeled = load_data_file("test/resources/test.csv")

    def test_embed_unlabeled(self):
        g = TextPairEmbeddingGenerator(self.unlabeled, batch_size=4)
        self.assertEqual(9, len(g))
        self.assertEqual("TextPairEmbeddingGenerator: 9 samples, batch size 4, maximum tokens 20", str(g))
        self.assertEqual(None, g.classes)
        self.assertEqual(3, g.batches_per_epoch)
        two_epochs = list(islice(g(), 2 * g.batches_per_epoch))
        self._validate_unlabeled_batches(two_epochs, g.batches_per_epoch, 20, [4, 4, 1] * 2)

    def test_embed_labeled(self):
        g = TextPairEmbeddingGenerator(self.labeled, batch_size=32)
        assert_array_equal([0, 1], g.classes)
        self.assertEqual(100, len(g))
        self.assertEqual("TextPairEmbeddingGenerator: 100 samples, classes [0, 1], batch size 32, maximum tokens 40",
                         str(g))
        self.assertEqual(4, g.batches_per_epoch)
        two_epochs = list(islice(g(), 2 * g.batches_per_epoch))
        self._validate_labeled_batches(two_epochs, g.batches_per_epoch, 40, [32, 32, 32, 4] * 2)

    def test_embed_labeled_specified_maximum_tokens(self):
        g = TextPairEmbeddingGenerator(self.labeled, batch_size=32, maximum_tokens=10)
        self.assertEqual(100, len(g))
        self.assertEqual("TextPairEmbeddingGenerator: 100 samples, classes [0, 1], batch size 32, maximum tokens 10",
                         str(g))
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
        data = load_data_file("test/resources/train.csv")
        n = int(0.8 * len(data))
        self.train = data[:n]
        self.validate = data[n:]
        self.test = load_data_file("test/resources/test.csv")
        self.temporary_directory = tempfile.mkdtemp()
        self.model_directory = os.path.join(self.temporary_directory, "model")

    def test_properties(self):
        model = TextPairClassifier.create(2, 40, 300, 128, 0.5, False)
        self.assertEqual(40, model.maximum_tokens)
        self.assertEqual(300, model.embedding_size)
        self.assertEqual(128, model.lstm_units)
        self.assertEqual(0.5, model.dropout)
        self.assertEqual(2, model.classes)
        self.assertEqual(False, model.bidirectional)
        model = TextPairClassifier.create(2, 40, 300, 128, 0.5, True)
        self.assertEqual(True, model.bidirectional)

    def test_stringification(self):
        model = TextPairClassifier.create(2, 40, 300, 128, 0.5, False)
        self.assertEqual(
            "TextPairClassifier(classes = 2, LSTM units = 128, maximum tokens = 40, embedding size = 300, "
            "dropout = 0.50)",
            repr(model))
        model = TextPairClassifier.create(2, 40, 300, 128, None, False)
        self.assertEqual(
            "TextPairClassifier(classes = 2, LSTM units = 128, maximum tokens = 40, embedding size = 300, No dropout)",
            repr(model))
        model = TextPairClassifier.create(2, 40, 300, 128, 0.5, True)
        self.assertEqual(
            "TextPairClassifier(bidirectional, classes = 2, LSTM units = 128, maximum tokens = 40, embedding size = "
            "300, dropout = 0.50)",
            repr(model))

    # noinspection PyUnresolvedReferences
    def test_train_predict_score(self):
        # Train
        model, history = TextPairClassifier.train(self.train, False, 128, 2,
                                                  dropout=0.5, maximum_tokens=30,
                                                  validation_data=self.validate,
                                                  model_directory=self.model_directory)
        self.assertEqual(
            "TextPairClassifier(classes = 2, LSTM units = 128, maximum tokens = 30, embedding size = 300, "
            "dropout = 0.50)",
            repr(model))
        self.assertIsInstance(model, TextPairClassifier)
        self.assertIsInstance(history, TrainingHistory)
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.info.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        # Predict
        predictions = model.predict(self.test)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(self.test), len(predictions))
        assert_array_equal([0, 1], predictions.columns)
        self.assertTrue((predictions >= 0).all)
        self.assertTrue((predictions <= 1).all)
        row_marginals = predictions.sum(axis=1)
        assert_allclose(row_marginals, ones(len(self.test)), rtol=1e-04)
        # Score
        scores = model.score(self.train)
        self.assertIsInstance(scores, list)
        self.assertTrue(2, len(scores))
        self.assertEquals({"loss", "acc"}, set(s[0] for s in scores))
        self.assertGreaterEqual(scores[0][1], 0)
        self.assertLessEqual(scores[0][1], 1)
        self.assertGreaterEqual(scores[1][1], 0)
        self.assertLessEqual(scores[1][1], 1)

    def test_train_no_validation(self):
        model, history = TextPairClassifier.train(self.train.head(20), False, 128, 1, dropout=0.5,
                                                  maximum_tokens=30, model_directory=self.model_directory)
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.info.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        self.assertIsInstance(model, TextPairClassifier)
        self.assertIsInstance(history, TrainingHistory)

    def test_train_no_model_directory(self):
        model, history = TextPairClassifier.train(self.train.head(20), False, 128, 1, dropout=0.5, maximum_tokens=30)
        self.assertIsInstance(model, TextPairClassifier)
        self.assertIsInstance(history, TrainingHistory)

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


class TestCommandLine(TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.model_directory = os.path.join(self.temporary_directory, "model")

    def test_no_arguments(self):
        actual = main_function_output([])
        self.assertEqual(
            "usage: bisemantic [-h] [--version] [--log LOG]\n                  " +
            "{train,continue,predict,score,cross-validation} ...\n", actual)

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

    def test_train_predict_score(self):
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
        main_function_output(["score", self.model_directory, "test/resources/train.csv"])

    def test_train_predict_snli_format(self):
        snli_format = [
            "--not-comma-delimited",
            "--index-name", "pairID",
            "--text-1-name", "sentence1",
            "--text-2-name", "sentence2",
            "--label-name", "gold_label",
            "--invalid-labels", "'-'"
        ]
        main_function_output(["train", "test/resources/snli.csv",
                              "--validation-set", "test/resources/snli.csv",
                              "--units", "64",
                              "--dropout", "0.5",
                              "--epochs", "2",
                              "--model", self.model_directory
                              ] + snli_format)
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        training_history_filename = os.path.join(self.model_directory, "training-history.json")
        self.assertTrue(os.path.isfile(training_history_filename))
        training_history = TrainingHistory.load(training_history_filename)
        self.assertEqual("Training history, 1 runs", str(training_history))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.info.txt")))
        main_function_output(["predict", self.model_directory, "test/resources/snli.csv"] + snli_format)

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
