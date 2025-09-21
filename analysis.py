"""
Generate some plots for analysis
"""

import arviz as az
from pathlib import Path

result_file = (
    Path(__file__).parent.resolve() / "data" / "results" / "model_posterior.nc"
)
trace = az.from_netcdf(result_file)

# Key Questions: Which players have the highest overall scores?
# Are strategies noticeably different across players? Across expansions?
# Does Matt tuck cards more than Camille?
# Which strategy breakdowns lead to more wins?
