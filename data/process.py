"""
Process the raw scores data.

Realistically, polars is already set up for this sort of "chaining"  of operations
so it doesn't need to split up too much into separate functions.
"""

import logging
import polars as pl
from pathlib import Path
import os


logLevel = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=logLevel)
logger = logging.getLogger(__name__)


def input_csv_validation(df: pl.DataFrame) -> bool:
    """
    Validate the input CSV data to ensure it contains the expected columns.
    """
    # use a set here, because by convention, we're NEVER going to refer to the order. And no duplicates!
    logger.info("Validating input CSV columns.")

    expected_columns = {
        "Game",
        "Player",
        "Expansions",
        "Birds",
        "Bonus Cards",
        "Food on Cards",
        "Eggs",
        "Tucked Cards",
        "Nectar",
        "Duet Tokens",
        "Total",
    }
    actual_columns = set(df.columns)
    return expected_columns.issubset(actual_columns)


if __name__ == "__main__":
    data_dir = (
        Path(__file__).parent.parent / "data"
    )  # assume here that the raw data is present in data
    score_file = data_dir / "raw" / "scores.csv"

    logging.info(f"Reading raw score data from {score_file}")
    scores = pl.read_csv(score_file)

    assert input_csv_validation(scores), (
        "Input CSV does not contain the expected columns."
    )

    # For each game, we want to determine the winner, based soley on the maxium number of points
    max_score = scores.group_by("Game").agg(pl.col("Total").max().alias("max_score"))

    # create the actual scores.
    logger.info("Abbreviating expansion names and performing data cleaning.")
    scores = (
        scores.join(max_score, on="Game")
        .with_columns(
            # in the event of ties, we're all winners!
            (pl.col("Total") == pl.col("max_score")).alias("winner"),
            # Standard Expansion Names
            pl.when(pl.col("Expansions").eq("EE"))
            .then(pl.lit("Europe"))
            .when(pl.col("Expansions").eq("AE"))
            .then(pl.lit("Asia"))
            .when(pl.col("Expansions").eq("EE,OE"))
            .then(pl.lit("Oceania"))
            .when(pl.col("Expansions").eq("EE,OE,AE"))
            .then(pl.lit("All"))
            .otherwise(pl.lit("Other"))
            .alias("Expansions"),
        )
        .select(pl.exclude("max_score"))
        .with_columns(
            (
                (pl.col(col) / pl.col("Total")).alias(
                    f"{col.lower().replace(' ', '_')}_pct"
                )
                for col in [
                    "Birds",
                    "Bonus Cards",
                    "Food on Cards",
                    "Eggs",
                    "Tucked Cards",
                    "Nectar",
                    "Duet Tokens",
                ]
            )
        )
    )
    (processed_dir := data_dir / "processed").mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing processed data to {processed_dir / 'processed_scores.csv'}")
    scores.write_csv(processed_dir / "processed_scores.csv")
