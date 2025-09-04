"""
Data preprocessing pipeline for the model. The main goal is to convert to 
categorical variables, otherwise we're leaving everything alone. But also 
convert to use JAX arrays instead of numpy arrays. 
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
import jax.numpy as jnp
import polars as pl
from pathlib import Path

# the columns are pretty much set in stone for better or worse.
data_transformer = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), ["Expansions", "Player"]),
        (
            "pass",
            "passthrough",
            [
                "total_points",
                "birds_pct",
                "bonus_cards_pct",
                "food_on_cards_pct",
                "eggs_pct",
                "tucked_cards_pct",
                "nectar_pct",
                "duet_tokens_pct",
                "winner"
            ],
        ),
    ]
)

preprocessor = Pipeline([
    ("data_transformer", data_transformer),
    ("converter", FunctionTransformer(lambda x: jnp.array(x), validate=False)),
])


if __name__ == "__main__":
    
    processed_data_path = Path(__file__).parent / "processed" / "processed_scores.csv"
    processed_scores = pl.read_casv(processed_data_path)
    
    # Fit and transform the data
    fit_data = preprocessor.fit_transform(processed_scores)

    

