import numpy as np
from jax import numpy as jnp


def test_data_preprocessing(preprocessed_data):
    assert isinstance(preprocessed_data, jnp.ndarray), (
        "Preprocessed data should be a JAX numpy array."
    )


def test_ordinal_encoding(preprocessed_data):
    # this test is purposefully brittle - we need to ensure that the first
    # column is expansions and the second is players.
    expansions, players = preprocessed_data[:, :2].T
    assert (np.unique(expansions) == [0, 1, 2, 3]).all(), (
        "There should be 4 expansions, labelled 0-3."
    )
    # if you change the data and add more players, that should be ok.
    assert len(np.unique(players)) > 10, (
        "There should be at least 10 players, though more is ok."
    )


def test_nonzero_shape(preprocessed_data):
    assert preprocessed_data.shape[0] > 0 and preprocessed_data.shape[1] > 2, (
        "Preprocessed data should have non-zero rows"
    )
