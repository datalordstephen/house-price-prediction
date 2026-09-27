from src.models import MODEL_COLORS, MODEL_REGISTRY, slugify


def test_registry_has_seven_uniquely_named_models():
    assert len(MODEL_REGISTRY) == 7
    assert len(set(MODEL_REGISTRY)) == 7
    assert next(iter(MODEL_REGISTRY)) == "Baseline (Mean)"


def test_every_model_has_a_colour():
    assert set(MODEL_COLORS) == set(MODEL_REGISTRY)


def test_each_model_builds_a_fittable_estimator():
    for build_model in MODEL_REGISTRY.values():
        estimator = build_model()
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")


def test_slugify():
    assert slugify("Baseline (Mean)") == "baseline_mean"
    assert slugify("Neural Net (MLP)") == "neural_net_mlp"
    assert slugify("Linear Regression") == "linear_regression"
    assert slugify("Gradient Boosting") == "gradient_boosting"
