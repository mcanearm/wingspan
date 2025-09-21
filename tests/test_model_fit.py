import arviz as az


def test_run_model(mcmc):
    # test that the model runs without error
    samples = mcmc.get_samples()
    assert "points_total" in samples
    assert "theta" in samples


def test_valid_numpyro(mcmc):
    # runs without error and prodiuces an ArviZ InferenceData object
    assert az.from_numpyro(mcmc)


def test_convergence(mcmc):
    # Assert that key variables exist
    trace = az.from_numpyro(mcmc)
    assert az.summary(trace)["r_hat"].max() <= 1.01
