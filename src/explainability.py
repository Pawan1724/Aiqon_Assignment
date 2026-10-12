"""
explainability.py  –  Model explanations (SHAP + permutation importance).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import CONFIG, logger

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("SHAP not available; falling back to permutation importance only.")

from sklearn.inspection import permutation_importance


# ------------------------------------------------------------------ #
#  Permutation importance                                             #
# ------------------------------------------------------------------ #

def compute_permutation_importance(
    model,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    feature_names: list[str] | None = None,
    n_repeats: int = 10,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Compute permutation importance and return as a sorted DataFrame.
    """
    seed = seed or CONFIG["project"]["random_seed"]
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=seed, scoring="neg_mean_absolute_error",
    )
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return df


# ------------------------------------------------------------------ #
#  SHAP explanations                                                  #
# ------------------------------------------------------------------ #

def compute_shap_values(
    model,
    X_test: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    max_samples: int = 100,
) -> dict[str, Any] | None:
    """
    Compute SHAP values for the model using TreeExplainer or KernelExplainer.

    Returns dict with:
        shap_values : np.ndarray
        expected_value : float
        feature_names : list[str]
    Or None if SHAP is unavailable / fails.
    """
    if not _SHAP_AVAILABLE:
        return None

    try:
        # Get the final estimator from the pipeline
        if hasattr(model, "named_steps"):
            preprocessor = model.named_steps.get("preprocessor")
            estimator = model.named_steps.get("model")
            if preprocessor is not None:
                X_transformed = preprocessor.transform(X_test)
            else:
                X_transformed = X_test
        else:
            estimator = model
            X_transformed = X_test

        if isinstance(X_transformed, pd.DataFrame):
            X_transformed = X_transformed.values

        # Limit samples
        if X_transformed.shape[0] > max_samples:
            idx = np.random.RandomState(42).choice(
                X_transformed.shape[0], max_samples, replace=False
            )
            X_transformed = X_transformed[idx]

        # Try TreeExplainer first
        try:
            explainer = shap.TreeExplainer(estimator)
            sv = explainer.shap_values(X_transformed)
            ev = explainer.expected_value
        except Exception:
            # Fallback to KernelExplainer with subsample
            bg = shap.kmeans(X_transformed, min(10, len(X_transformed)))
            explainer = shap.KernelExplainer(estimator.predict, bg)
            sv = explainer.shap_values(X_transformed, nsamples=100)
            ev = explainer.expected_value

        return {
            "shap_values": sv,
            "expected_value": ev,
            "feature_names": feature_names or [f"f_{i}" for i in range(X_transformed.shape[1])],
            "X_display": X_transformed,
        }

    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None


# ------------------------------------------------------------------ #
#  Plotting helpers                                                    #
# ------------------------------------------------------------------ #

def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance (Permutation)",
    save_path: str | None = None,
    top_n: int = 15,
) -> str | None:
    """Bar chart of feature importance. Returns save path."""
    df_top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_top) * 0.35)))
    ax.barh(
        df_top["feature"][::-1],
        df_top["importance_mean"][::-1],
        xerr=df_top["importance_std"][::-1],
        color="#4C72B0", edgecolor="white",
    )
    ax.set_xlabel("Mean |ΔMAE| (permutation)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Feature importance plot saved to %s", save_path)
        return save_path
    plt.close(fig)
    return None


def plot_shap_summary(
    shap_result: dict,
    save_path: str | None = None,
) -> str | None:
    """SHAP beeswarm summary plot."""
    if not _SHAP_AVAILABLE or shap_result is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_result["shap_values"],
        features=shap_result["X_display"],
        feature_names=shap_result["feature_names"],
        show=False,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP summary plot saved to %s", save_path)
        return save_path
    plt.close("all")
    return None


def explain_single_prediction(
    model,
    X_single: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Return a human-readable explanation for one prediction.

    Computes exact model feature attribution in minutes.
    """
    prediction = model.predict(X_single)[0]

    explanation: dict[str, Any] = {
        "predicted_delay_min": round(float(prediction), 1),
        "top_factors": [],
        "note": "",
    }

    # If linear / Ridge pipeline, compute exact mathematical feature contributions
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        estimator = model.named_steps.get("model")
        if estimator is not None and hasattr(estimator, "coef_") and preprocessor is not None:
            try:
                Xt = preprocessor.transform(X_single)
                if hasattr(Xt, "toarray"):
                    Xt = Xt.toarray()
                x_row = Xt[0]
                coefs = estimator.coef_
                out_names = preprocessor.get_feature_names_out()

                contribs = []
                for feat_raw, val in zip(out_names, x_row * coefs):
                    f_clean = feat_raw.replace("num__", "").replace("cat__", "")
                    contribs.append((f_clean, float(val)))

                # Sort by absolute impact
                contribs.sort(key=lambda x: abs(x[1]), reverse=True)

                for feat, val in contribs[:6]:
                    if abs(val) >= 0.05:
                        direction = "increases" if val > 0 else "decreases"
                        explanation["top_factors"].append({
                            "feature": feat,
                            "contribution_min": round(abs(val), 1),
                            "direction": direction,
                        })

                explanation["note"] = (
                    "Exact feature attribution calculated from trained regression coefficients and standardized features."
                )
                return explanation
            except Exception as e:
                logger.debug("Direct linear attribution fallback: %s", e)

    # Fallback to SHAP
    shap_res = compute_shap_values(model, X_single, feature_names, max_samples=1)
    if shap_res is not None and shap_res["shap_values"] is not None:
        sv = shap_res["shap_values"]
        if sv.ndim > 1:
            sv = sv[0]
        contributions = list(zip(feature_names, sv.tolist()))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, val in contributions[:5]:
            direction = "increases" if val > 0 else "decreases"
            explanation["top_factors"].append({
                "feature": feat,
                "contribution_min": round(abs(val), 1),
                "direction": direction,
            })
        explanation["note"] = (
            "Feature contributions show model attribution."
        )
    else:
        explanation["note"] = (
            "Global feature importance reflects overall trained model weights."
        )

    return explanation
