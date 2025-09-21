# Wingspan Stat Tracking and Modeling

## Introduction

I am an avid Wingspan player. While living in Ann Arbor in our first home, my 
wife and I had a weekly tradition. Each Saturday, we would brew coffee and 
rotate our couch to face the front yard. Directly in front of our living
room window was a magnolia tree from which we hung two birdfeeders. We would 
watch chickadees, tufted titmouses, cardinals, blue jays, sparrows, 
and finches all feed from our birdfeeders while we played the game.

Uncontent with simply enjoying these beautiful memories, I decided it would
be fun to track our scores from our games and see which strategies are optimal. Despite
relatively simple rules, there is an incredible amount of variety and strategy to the game, due
to the combinatorics around birds, when they are played, and the many avenues by which players can score
points.  Given the wins, losses, and point distributions between the two of us 
(and guests at other points), I set up a simplified Bayesian model to 
estimate which strategies are best for which expansions.

## The Game

[Wingspan](https://stonemaiergames.com/games/wingspan/) was published on March 8th, 2019, by Stonemaier games.
The game proceeds as follows (in broad strokes):

1) You have 26 turns over the course of the game. In four rounds, you start with 8 turns, then 7, then 6, and finally 5.
2) There are four possible moves on any turn - play a bird card, gain food, gain eggs, or draw cards. The last three
actions correspond to "activating" a habitat, and birds played in previous turns may or may not have effects to improve
the number of resources you receive from a given habitat.
3) At the end of the game, you accumulate points based on a few criteria:
   1) The point value of the birds you've played
   2) The number of eggs on your board
   3) The number of tucked cards (a special power some birds have)
   4) The number of food tokens cached on bird cards
   5) The value of bonus point cards
   6) End of round goals
   7) Nectar habitat points (5 or 2 for 1st or 2nd place, Oceania always and sometimes Asia)
   8) Duet token goals (Asia Duet mode only)

Generally, you should build an "engine" to generate points in rounds 1-2, run the engine in round 3, and 
finally accumulate as many points as possible in round 4 through whatever means are available. There is an element
of stochasticity in each strategy however, and so the question becomes: "which strategy is most likely to yield the
maximum number of points, especially relative to the other players?" 

This analysis in particular focuses on using a Bayesian hierarchical modeling framework with simplified models
to jointly estimate multiple parameters of interest, such as point distribution by category and expected point total, and 
point variance. I utilize a single-layer Multinomial-Dirichlet model, which works well in practice. Certain
complexities around scoring are also handled via the use of a masking function, which is detailed further in `model.py`
for those interested.

This package is not yet mature enough to generate an predict an actual winning strategy - it is mostly limited to descriptive statistics on which strategies yield the most points. The framework is in place to add simple models, however, and it is reasonable to estimate both average point allocation and point score based on a Dirichlet-Multinomial model.


## Running the analysis

The package comes with the data required. It's a short file and minimal 
preprocessing is required. Still, the process is as follows: 
`./data/process.py -> ./model.py -> ./analysis.py`. The outputs of the analysis 
are deposited in `./data/results/`, which contains the model posterior, and
forest plots summarizing some distributions of interest around questions that
inspired this analysis are written to `./data/results/figures`. Most simply, 
just run `sh run_analysis.py` to do all of this in order.

## Requirements

The requirements are relatively light, but the most notable and potentially 
fussy step is the installation of `Numpyro` and `JAX`. It is assumed that you will
use the `pip` package manager to rerun the analysis. `conda` should work as well,
but is not explicity supported through an environment file.

Still, to get everything you need, the standard `pip install -r requirements.txt` 
should manage. If you would like to run the provided notebook interactively,
run tests, you may install those packages using the ironically named `optional-requirements.txt`. 

Also included is a pre-commit hook. Contributions to the package utilize the `ruff`
package for formatting (per course requirements), and the pre-commit hook enforces
this formatting on all code commits. To utilize, run
```
pip install -r optional-requirements.txt
pre-commit install
```
