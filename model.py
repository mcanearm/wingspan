"""
Model module for the application.
"""

from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import polars as pl
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Data preprocessing pipeline for the model. The main goal is to convert
# categorical variables to ordinal ones; otherwise, we're leaving everything alone.
# Also convert to use JAX arrays instead of numpy arrays.
# the columns are pretty much set in stone for better or worse.
data_transformer = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), ["Expansions", "Player"]),
        (
            "pass",
            "passthrough",
            [
                "Birds",
                "Bonus Cards",
                "End of Round Goals",
                "Eggs",
                "Food on Cards",
                "Tucked Cards",
                "Nectar",
                "Duet Tokens",
            ],
        ),
    ]
)

preprocessor = Pipeline(
    [
        ("data_transformer", data_transformer),
        ("converter", FunctionTransformer(lambda x: jnp.array(x), validate=False)),
    ]
)


def model(expansions, players, counts):
    """
    This is the main model for the Wingspan scoring data. It is hierarchical across
    two dimensions (player and game expansion), but these dimensions are not nested, and
    so the tuple of (player, expansion) is the key dimension. This gives a three
    dimensional array of shape (n_players, n_expansions, n_categories), and we use
    this representation to model the point distributions. Everything else after
    the multinomial distribution is fairly standard.

    The expected observation level is a (game, player) tuple. For now, the game
    is completely ignored, but in future analyses, it could be useful to know
    whether certain games were more competitive than others and how that affects
    the overall strategy. But for now, we treat the (player, game) tuples
    as interchangeable.

    params:
    expansions: array (n_obs,), ordinal encoding of expansion type
    players: array (n_obs,), ordinal encoding of player
    counts: array (n_obs, n_categories), raw point counts across n_obs and n_categories

    returns:
    None
    """
    N, K = counts.shape
    total = jnp.sum(counts, axis=1)
    n_players = len(np.unique(players))

    # there are always 4, but maybe one day...
    n_expansions = len(np.unique(expansions))

    # Global base distribution across categories. We simulate the distribution
    # of points across categories as a percentage, and separately, model the
    # total number of points scored. The "plate" syntax repeats the sampling
    # process within a dimension, and gives a more readable way to express
    # hierarchical models.

    # these are the global parameters
    phi = numpyro.sample("phi", dist.Dirichlet(jnp.ones(K)))
    point_mean = numpyro.sample("point_mean", dist.Normal(70.0, 10))
    point_sigma = numpyro.sample("point_sigma", dist.HalfNormal(15.0))

    # each category has its own concentration parameter
    with numpyro.plate("categories", K):
        log_eta = numpyro.sample("log_eta", dist.Normal(0.0, 1.0))
        eta = jnp.exp(log_eta)

    # Each expansion does not have the same valid point categories. For example, only
    # the Asian expansion in two player mode has the "Duet Tokens" category. Therefore,
    # to jointly estimate ALL expansion categories at once, we need to make the concentration
    # of certain point categories go to zero.
    point_mask = jnp.array(
        [
            # Birds, Bonus Cards, End of Round Goals, Eggs, Food on Cards, Tucked Cards, Nectar, Duet Tokens
            [1, 1, 1, 1, 1, 1, 1, 1],  # All
            [1, 1, 0, 1, 1, 1, 1, 1],  # Asia
            [1, 1, 1, 1, 1, 1, 0, 0],  # Europe
            [1, 1, 1, 1, 1, 1, 1, 0],  # Oceania
        ],
    )
    conc = eta * phi * point_mask + 0.5

    # each player and expansion tuple has its own distribution of points AND point total
    with numpyro.plate("expansions", n_expansions):
        with numpyro.plate("players", n_players):
            point_lambda = numpyro.sample(
                "points_total", dist.Normal(point_mean, point_sigma)
            )
            theta = numpyro.sample("theta", dist.Dirichlet(conc))
            numpyro.deterministic("points_category", theta * point_lambda[:, :, None])  # type: ignore

    point_eps = numpyro.sample("point_eps", dist.HalfNormal(5.0))
    with numpyro.plate("obs", N):
        numpyro.sample(
            "obs_counts",
            dist.Multinomial(total_count=total, probs=theta[players, expansions]),  # type: ignore
            obs=counts,
        )
        numpyro.sample(
            "obs_total",
            dist.Normal(point_lambda[players, expansions], point_eps),  # type: ignore
            obs=total,
        )


if __name__ == "__main__":
    """
    With the model specified, we can run the script to actually fit the model
    to our data.
    """

    # assume a 4 core machine for now. Maybe not the best, but... come on.
    # Smart fridges have at least 4 cores. And you don't need a GPU.
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)

    processed_data = (
        Path(__file__).parent.resolve() / "data" / "processed" / "processed_scores.csv"
    )
    try:
        processed_data = pl.read_csv(processed_data)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Processed data not found, please run `python data/process.py` first."
        )

    fit_data = preprocessor.fit_transform(processed_data)

    # use the default NUTS parameters. It worked fine, but you can tune it
    # if you want.
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4)

    key = jax.random.key(481)

    # actually important we don't use jnp here, since that will screw
    # up the model fitting. jnp arrays have "tracer" behavior that means
    # the shape needs to be known at runtime/compile time otherwise the program
    # fails. However, use jnp for the count data, since that shape is static.
    expansion_labels, players = np.array(fit_data[:, :2].T, dtype=jnp.int32)
    points = fit_data[:, 2:]

    # Run the actual model and save the results as a NetCDF file, since the
    # trace has arrays of varying shapes due to the hierarchical structure.
    mcmc.run(key, expansions=expansion_labels, players=players, counts=points)

    expansion_labels, player_names = data_transformer.transformers_[0][1].categories_
    point_categories = [
        cat.replace("pass__", "")
        for cat in data_transformer.get_feature_names_out()[2:]
    ]
    trace = az.from_numpyro(
        mcmc,
        coords={
            "players": player_names,
            "expansions": expansion_labels,
            "point_categories": point_categories,
        },
        dims={
            "phi": ["point_categories"],
            "point_totals": ["point_categories"],
            "theta": ["players", "expansions", "point_categories"],
            "points_total": ["players", "expansions"],
            "points_category": ["players", "expansions", "point_categories"],
        },
    )
    (result_dir := Path(__file__).parent.resolve() / "data" / "results").mkdir(
        parents=True, exist_ok=True
    )

    az.to_netcdf(trace, result_dir / "model_posterior.nc")
