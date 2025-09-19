import pytest
import pandas as pd


@pytest.fixture
def input_data():
    return pd.read_csv("./data/raw/scores.csv")
