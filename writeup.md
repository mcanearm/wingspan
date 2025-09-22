## Original State of Analysis

The original state was pretty bare bones. This was a project I did on a whim
and without any real objective. I had completeed a minor analysis previously, but
never saved any of it in Git. In fact, I also completely deleted my prior model so
I had to spend some time actually recoding a new one from scratch. The data 
processing pipeline was in place, however, and so I was able to reuse that fully.

## Biggest Challenges in the transformation

Most things were completely separated into cells, and the analysis was very simple
to begin with. The hardest part was probably the inclination to keep poking 
around the edges with changes, but also writing meaningful tests. I wanted to write
unit tests that were useful, but because I chose a really simple script, any tests
waste effort on re-testing basic functionality in other packages OR are kind of slow.
I decided it was better to have slow and useful tests based on the model fitting than
fast tests to regurgitate basic `jax`, `numpyro`, or `xarray` functions.

## Which improvements had the most impact on reproducibility?

Repdroducibility was most improved by adding the `analysis` script. The reason
the notebook was disorganized was that I kept changing cells, re-running, and 
seeing the result. I had some basic questions in mind that I wanted to answer, so
putting them into an analysis script sort of sets those in stone. I also left
an interactive notebook in place for more ad hoc questions.

I also really liked adding the pre-commit hook for the `ruff` formatter. It meant that
whenever I made a commit, whatever file I touched was automatically formatted.

## What would you do differently?

I would like to have been a bit more careful with my Git history. I tend
to lots of small commits and many branches to keep things separate. I like this
better than doing massive commits, especially when you can squash several commits
on a branch into a single commit. However, the branch history can be difficult 
to parse.


## How long did each major component take to implement?

* Model - 2-3 hours.
* Testing - 1 hour
* Documentation - 1-2 hours
* Processing pipeline - 1-2 hours
* Git setup - 5 minutes
