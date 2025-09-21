import jax
import numpy as np
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
import pytest
import pandas as pd
from model import preprocessor, model
import numpyro


# The use of fixtures makes this slower, but ensures that we aren't changing
# the data from test to test. In addition, we utilize the MCMC in a single file,
# so tests on data processing and model fitting can be reun independently to
# save time.


@pytest.fixture
def input_data():
    return pd.read_csv("./data/raw/scores.csv")


@pytest.fixture
def preprocessed_data(input_data):
    # fixture to preprocess the input data once for all tests in this module.
    transformed_data = preprocessor.fit_transform(input_data)
    return transformed_data


@pytest.fixture
def mcmc(preprocessed_data):
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)

    kernel = NUTS(model)
    # this makes the tests slow, but ensures that the model gives you valid
    # outputs.
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4)

    key = jax.random.key(0)
    expansion_labels, players = np.array(preprocessed_data[:, :2].T, dtype=jnp.int32)
    counts = preprocessed_data[:, 2:]

    # If the model signature changes, this test will break.
    mcmc.run(
        key,
        expansions=expansion_labels,
        players=players,
        counts=counts,
    )
    return mcmc
