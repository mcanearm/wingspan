"""
Generate some plots for anlaysis and so answer some key questions.
"""

import logging
import os
from pathlib import Path

import arviz as az
from matplotlib import pyplot as plt

logLevel = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=logLevel)
logger = logging.getLogger(__name__)

# Set to true to see the figures pop up as they are generated.
show_figures = False


try:
    cwd = Path(__file__).parent.resolve()
except NameError:
    cwd = Path().resolve()  # when running in an interactive environment
result_file = cwd / "data" / "results" / "model_posterior.nc"
(figure_dir := cwd / "data" / "results" / "figures").mkdir(parents=True, exist_ok=True)


# spit out the basic diagnostic trace plots for a high level diagnostic
logger.info("Reading model trace.")
trace = az.from_netcdf(result_file)

logger.info("Generating plots for analysis...")
logger.debug("Geenerating trace plot.")
az.plot_trace(trace)
plt.tight_layout()
plt.savefig(figure_dir / "trace_plots.png")
if show_figures:
    plt.show()

# This will happen a few times, but we simply have to rearrange the dimensions
# such that the things we want to compare are next to each other.
trace.posterior["points_total"] = trace.posterior["points_total"].transpose(  # type: ignore
    "draw", "chain", "expansions", "players"
)

# Key Questions: Between Camille and I, who scores more points, and over what
# expansions?
logger.debug("Generating total points comparison plot.")
az.plot_forest(
    trace,
    var_names=["points_total"],
    combined=True,
    hdi_prob=0.95,
    figsize=(8, 6),
    kind="forestplot",
    coords={"players": ["Matt", "Camille"]},
)
plt.tight_layout()
plt.savefig(figure_dir / "total_points_for_camille_and_matt.png")
if show_figures:
    plt.show()

trace.posterior["theta"] = trace.posterior["theta"].transpose(  # type: ignore
    "draw", "chain", "players", "point_categories", "expansions"
)

# Are strategies noticeably different across expansions, considering all players?
logger.debug("Generating strategy comparison by expansion plot.")
az.plot_forest(
    trace,
    var_names=["theta"],
    combined=True,
    combine_dims={"players"},
    hdi_prob=0.95,
    figsize=(8, 6),
    kind="forestplot",
    coords={
        "point_categories": [
            "Birds",
            "Eggs",
            "Bonus Cards",
            "End of Round Goals",
            "Food on Cards",
            "Tucked Cards",
            "Duet Tokens",
            "Nectar",
        ],
    },
)
plt.xlabel("$\\theta$")
plt.ylabel("Point Category")
plt.tight_layout()
plt.savefig(figure_dir / "strategy_by_expansion.png")
if show_figures:
    plt.show()


# Are strategies noticeably different across expansions, considering just Camille and myself??
logger.debug("Generating strategy comparison by expansion for Matt and Camille plot.")
az.plot_forest(
    trace,
    var_names=["theta"],
    combined=True,
    combine_dims={"players"},
    hdi_prob=0.95,
    figsize=(8, 6),
    kind="forestplot",
    coords={
        "point_categories": [
            "Birds",
            "Eggs",
            "Bonus Cards",
            "End of Round Goals",
            "Food on Cards",
            "Tucked Cards",
            "Duet Tokens",
            "Nectar",
        ],
        "players": ["Matt", "Camille"],
    },
)
plt.xlabel("$\\theta$")
plt.ylabel("Point Category")
plt.tight_layout()
plt.savefig(figure_dir / "strategy_by_expansion_matt_and_camille.png")
if show_figures:
    plt.show()


# Does Matt rely on card tucking more than Camille, in either raw points
# or proportionally?
logger.debug("Generating tucked cards distribution plot for Matt and Camille plot.")
trace.posterior["theta"] = trace.posterior["theta"].transpose(  # type: ignore
    "draw", "chain", "expansions", "point_categories", "players"
)
total_proportional_dist = az.plot_forest(
    trace,
    var_names=["theta"],
    coords={
        "players": ["Matt", "Camille"],
        "point_categories": ["Tucked Cards"],
    },
    figsize=(8, 6),
    combined=True,
    hdi_prob=0.95,
)
plt.xlabel("$\\theta$")
plt.ylabel("Expansion,Player")
plt.title("Proportional Points from Tucked Cards HDI")
plt.tight_layout()
plt.savefig(figure_dir / "tucked_cards_dist.png")
if show_figures:
    plt.show()
