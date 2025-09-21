import arviz as az


def test_run_model(mcmc):
    # test that the model runs without error and produces output of expected shape.
    samples = mcmc.get_samples()
    assert (
        trace := az.from_numpyro(mcmc)
    )  # runs without error and prodiuces an ArviZ InferenceData object

    # Assert that key variables exist
    assert "points_total" in samples
    assert "theta" in samples
    assert az.summary(trace)["r_hat"].max() <= 1.01
