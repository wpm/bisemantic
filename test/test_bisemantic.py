import sys
from io import StringIO
from unittest import TestCase

import pandas as pd

from bisemantic import load_data
from bisemantic.main import main


def main_function_output(args):
    sys.argv = ["bisemantic"] + args
    sys.stdout = s = StringIO()
    try:
        main()
    except SystemExit:
        pass
    sys.stdout = sys.__stdout__
    return s.getvalue()


class TestCommandLine(TestCase):
    def test_no_arguments(self):
        actual = main_function_output([])
        self.assertEqual("""usage: bisemantic [-h] [--version] [--log LOG] {train,predict} ...\n""", actual)

    def test_version(self):
        actual = main_function_output(["--version"])
        self.assertEqual("""bisemantic 1.0.0\n""", actual)


class TestPreprocess(TestCase):
    def test_load_data_with_null(self):
        actual = load_data("test/resources/data_with_null_values.csv")
        self.assertIsInstance(actual, pd.DataFrame)
        self.assertEqual(3, len(actual))


class TestEndtoEnd(TestCase):
    def test_end_to_end(self):
        main_function_output(["train", "test/resources/train.csv",
                              "--validation", "test/resources/train.csv"])
        main_function_output(["predict", "model", "test/resources/test.csv"])
