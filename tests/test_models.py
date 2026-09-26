from src.models import MODEL_REGISTRY, slugify


def test_registry_has_six_uniquely_named_models():
    assert len(MODEL_REGISTRY) == 6
    assert len(set(MODEL_REGISTRY)) == 6


def test_each_model_builds_a_fittable_estimator():
    for build_model in MODEL_REGISTRY.values():
        estimator = build_model()
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")


def test_slugify():
    assert slugify("Neural Net (MLP)") == "neural_net_mlp"
    assert slugify("Linear Regression") == "linear_regression"
    assert slugify("Gradient Boosting") == "gradient_boosting"
