import os
import shutil
import sys
import tempfile
from io import StringIO
from unittest import TestCase

import numpy as np
import pandas as pd
from keras.callbacks import History
from numpy.testing import assert_array_equal

from bisemantic import TextualEquivalenceModel
from bisemantic.console import main, data_file, fix_columns
from bisemantic.main import embed


class TestCommandLine(TestCase):
    def test_no_arguments(self):
        actual = main_function_output([])
        self.assertEqual("""usage: bisemantic [-h] [--version] [--log LOG] {train,predict} ...\n""", actual)

    def test_version(self):
        actual = main_function_output(["--version"])
        self.assertEqual("""bisemantic 1.0.0\n""", actual)


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
        train = fix_columns(self.train)
        assert_array_equal(["text1", "text2", "label"], train.columns)

    def test_fix_columns_with_invalid_column_name(self):
        self.assertRaises(ValueError, fix_columns, self.train, text_1_name="bogus")


class TestModel(TestCase):
    def setUp(self):
        data = data_file("test/resources/train.csv")
        n = int(0.8 * len(data))
        self.train = data[:n]
        self.validate = data[n:]
        self.test = data_file("test/resources/test.csv")

    def test_properties(self):
        model = TextualEquivalenceModel.create(40, 300, 128)
        self.assertEqual(40, model.maximum_tokens)
        self.assertEqual(300, model.embedding_size)
        self.assertEqual(128, model.lstm_units)

    def test_train_and_predict(self):
        model, history = TextualEquivalenceModel.train(self.train, 128, 2, self.validate)
        self.assertIsInstance(model, TextualEquivalenceModel)
        self.assertIsInstance(history, History)
        self.assertEqual("TextualEquivalenceModel(LSTM units = 128, maximum tokens = 40, embedding size = 300)",
                         str(model))
        predictions = model.predict(self.test)
        self.assertEqual((len(self.test),), predictions.shape)
        self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))


class TestSerialization(TestCase):
    def setUp(self):
        _, self.filename = tempfile.mkstemp('.h5')

    def test_serialization(self):
        model = TextualEquivalenceModel.create(40, 300, 128)
        model.save(self.filename)
        deserialized_model = TextualEquivalenceModel.load(self.filename)
        self.assertIsInstance(deserialized_model, TextualEquivalenceModel)
        self.assertEqual(model.maximum_tokens, deserialized_model.maximum_tokens)
        self.assertEqual(model.embedding_size, deserialized_model.embedding_size)
        self.assertEqual(model.lstm_units, deserialized_model.lstm_units)

    def tearDown(self):
        os.remove(self.filename)


class TestEmbedding(TestCase):
    def setUp(self):
        self.text_pairs = pd.DataFrame({
            "text1": ["horse mouse", "red black blue", "heart diamond", "triangle"],
            "text2": ["cat horse", "red green", "spade spade", "circle square rectangle"],
        })

    def test_embedding(self):
        embeddings, maximum_tokens = embed(self.text_pairs)
        self.assertIsInstance(embeddings, list)
        self.assertEqual(2, len(embeddings))
        self.assertEqual((4, 3, 300), embeddings[0].shape)
        self.assertEqual(3, maximum_tokens)

    def test_embedding_clip(self):
        embeddings, maximum_tokens = embed(self.text_pairs, maximum_tokens=2)
        self.assertIsInstance(embeddings, list)
        self.assertEqual(2, len(embeddings))
        self.assertEqual((4, 2, 300), embeddings[0].shape)
        self.assertEqual(2, maximum_tokens)


class TestEndToEnd(TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.model_directory = os.path.join(self.temporary_directory, "model")

    def test_end_to_end(self):
        main_function_output(["train", "test/resources/train.csv",
                              "--validation", "test/resources/train.csv",
                              "--units", "64",
                              "--model", self.model_directory])
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.h5")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "history.json")))
        main_function_output(["predict", self.model_directory, "test/resources/test.csv"])

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
