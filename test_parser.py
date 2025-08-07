import pandas as pd
import pytest
from custom_parsers.icici_parser import parse
import numpy as np

# Define the paths to the test files
PDF_PATH = "data/icici/icici sample.pdf"
CSV_PATH = "data/icici/result.csv"


def test_parser_output():
    """
    Tests that the generated parser's output matches the expected CSV.
    """
    # 1. Get the actual DataFrame from the generated parser
    df_generated = parse(PDF_PATH)

    # 2. Get the expected DataFrame from the provided CSV
    df_expected = pd.read_csv(CSV_PATH)

    # 3. Normalize both DataFrames for a reliable comparison
    # Convert numeric columns, coercing errors to NaN
    for col in ["Debit Amt", "Credit Amt", "Balance"]:
        df_generated[col] = pd.to_numeric(df_generated[col], errors="coerce")
        df_expected[col] = pd.to_numeric(df_expected[col], errors="coerce")

    # Reset index to ensure they align
    df_generated = df_generated.reset_index(drop=True)
    df_expected = df_expected.reset_index(drop=True)

    # 4. Assert that the DataFrames are equal
    # pandas.testing.assert_frame_equal is more rigorous than .equals()
    # It provides detailed output on mismatches.
    pd.testing.assert_frame_equal(df_generated, df_expected, check_dtype=False)
