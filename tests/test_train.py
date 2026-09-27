from src.models import MODEL_REGISTRY, slugify
from src.train import train_all


def test_train_all_produces_metrics_and_artifacts_for_every_model(tiny_df, tmp_path):
    metrics_df = train_all(df=tiny_df, artifacts_dir=tmp_path)

    assert len(metrics_df) == len(MODEL_REGISTRY)
    assert set(metrics_df["Model"]) == set(MODEL_REGISTRY)
    assert list(metrics_df.columns) == [
        "Model", "R2_log", "R2", "RMSE", "MAE", "MdAPE", "CV_R2_log_mean", "CV_R2_log_std",
    ]
    assert metrics_df["R2_log"].is_monotonic_decreasing
    assert (metrics_df["RMSE"] >= 0).all()
    assert (metrics_df["MAE"] >= 0).all()
    assert (metrics_df["MdAPE"] >= 0).all()
    assert (metrics_df["R2"] <= 1).all()
    assert (metrics_df["R2_log"] <= 1).all()

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    for name in metrics_df["Model"]:
        assert (tmp_path / f"{slugify(name)}.joblib").exists()
