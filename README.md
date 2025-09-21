# Wingspan Stat Tracker and Model Estimator

I am an avid Wingspan player. My wife and I had a tradition at a previous home that each 
Saturday, we would rotate our couch to face outside. Directly in front of our living
room window was a magnolia tree from which we hung two birdfeeders. We would make 
coffee and watch chickadees, tufted titmouses, cardinals, blue jays, sparrows, 
house finches, goldfinches, and squirrels all feed from our birdfeeders while we played Wingspan. 

Uncontent with simply enjoying these beautiful memories, I instead decided it would
be fun to track our scores from our games and see which strategies are optimal. Despite
relatively simple rules, there is an incredible amount of variety and strategy to the game,
and so given the wins and losses between the two of us (and guests at other points), we 
can set up a simplified Bayesian model to estimate which strategies are best for which expansions.
The game proceeds as follows (in broad strokes):

1) You have 26 turns over the course of the game. In four rounds, you start with 8 turns, then 7, then 6, and finally 5.
2) There are four possible moves on any turn - play a bird card, gain food, gain eggs, or draw cards. The last three
actions correspond to "activating" a habitat, and birds played in previous turns may or may not have effects to improve
the number of resources you receive from a given habitat.
3) At the end of the game, you accumulate points based on a few criteria:
   1) The point value of the birds you've played
   2) The number of eggs on your board
   3) The number of tucked cards (a special power some birds have)
   4) Cached food on cards
   5) The value of bonus points
   6) End of round goals
   7) Nectar habitat points (5 or 2 for 1st or 2nd place)
   8) Duet token goals (Asia Duet mode only)

Broadly speaking, you should build an "engine" to generate points in rounds 1-2, run the engine in round 3, and 
finally accumulate as many points as possible in round 4 through whatever means are available. There is an element
of stochasticity in each strategy however, and so the question becomes: "which strategy is most likely to yield the
maximum number of points, especially relative to the other players?" 

This package in particular focuses on using a Bayesian hierarchical modeling framework with simplified models
to jointly estimate multiple parameters of interest, such as optimal winning strategy, expected point total from a
given strategy, and overall player skill. 
