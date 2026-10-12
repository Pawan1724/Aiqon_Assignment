"""
prediction_service.py – Production-grade unified prediction service.

Combines ETA regression, delay classification, explainability,
and rule-based delay propagation into a single API.

No mocked or hardcoded data is returned. If models are not loaded,
prediction fields return None and the caller decides how to display that.
"""
from __future__ import annotations

import os
import sys
import types
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from src import CONFIG, logger
from src.propagation_engine import DelayPropagationEngine


def _patch_blocked_dlls() -> None:
    """
    Stub out optional sklearn native modules that Windows Application
    Control policies may block.

    This project uses Ridge and RandomForest only — the SVM / liblinear
    DLLs are never called, but their failed import can cascade and prevent
    joblib.load() from deserialising *any* sklearn pipeline.

    Must run before the first sklearn import.
    """
    if sys.platform != "win32":
        return
    _optional_native_modules = [
        "sklearn.svm._libsvm_sparse",
        "sklearn.svm._libsvm",
        "sklearn.svm._liblinear",
    ]
    for mod_name in _optional_native_modules:
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except (ImportError, OSError):
                sys.modules[mod_name] = types.ModuleType(mod_name)
                logger.debug("Patched blocked native module: %s", mod_name)


# Apply the patch immediately on module load, before any sklearn import.
_patch_blocked_dlls()


class PredictionService:
    """
    High-level service that loads trained ML models and provides
    predictions via a clean public API.

    If models are unavailable, prediction fields return ``None`` while
    the rule-based propagation engine remains fully operational.
    """

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or CONFIG["paths"]["models_dir"]
        if not os.path.isabs(self.model_dir):
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            self.model_dir = os.path.join(project_root, self.model_dir)

        self.eta_model = None
        self.eta_meta: dict = {}
        self.clf_model = None
        self.clf_meta: dict = {}
        self.propagation = DelayPropagationEngine()

        self._load_models()

    # -------------------------------------------------------------- #
    #  Model loading                                                   #
    # -------------------------------------------------------------- #

    def _load_models(self) -> None:
        """Load trained models from disk.  Logs errors but never crashes."""
        # --- ETA regression model ---
        try:
            from src.eta_model import load_model as load_eta
            self.eta_model, self.eta_meta = load_eta(self.model_dir)
            logger.info(
                "ETA model loaded: %s", self.eta_meta.get("model_name")
            )
        except FileNotFoundError:
            logger.warning(
                "ETA model not found at %s — run 'python scripts/train_models.py'.",
                self.model_dir,
            )
        except Exception as e:
            logger.error("Failed to load ETA model: %s", e, exc_info=True)

        # --- Delay classifier ---
        try:
            from src.delay_classifier import load_classifier
            self.clf_model, self.clf_meta = load_classifier(self.model_dir)
            logger.info(
                "Classifier loaded: %s", self.clf_meta.get("model_name")
            )
        except FileNotFoundError:
            logger.warning(
                "Classifier not found at %s — run 'python scripts/train_models.py'.",
                self.model_dir,
            )
        except Exception as e:
            logger.error("Failed to load classifier: %s", e, exc_info=True)

    @property
    def models_loaded(self) -> bool:
        """True when both ML models are available for inference."""
        return self.eta_model is not None and self.clf_model is not None

    # -------------------------------------------------------------- #
    #  Internal helpers                                                #
    # -------------------------------------------------------------- #

    @staticmethod
    def _build_features(shipment: dict[str, Any]) -> pd.DataFrame | None:
        """Build a single-row feature matrix from a shipment dict."""
        try:
            from src.feature_engineering import (
                NUMERIC_FEATURES,
                CATEGORICAL_FEATURES,
                engineer_features,
            )
            df = pd.DataFrame([shipment])
            df = engineer_features(df)
            num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
            cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
            return df[num_cols + cat_cols]
        except Exception as e:
            logger.error("Feature engineering failed: %s", e)
            return None

    def _run_propagation(self, shipment: dict[str, Any]) -> dict[str, Any]:
        """Run the rule-based delay propagation engine."""
        prop_result = self.propagation.propagate(
            scheduled_duration_min=shipment.get("scheduled_duration_min") or 0,
            departure_delay_min=shipment.get("departure_delay_min") or 0,
            weather_departure=shipment.get("weather_condition_departure") or "clear",
            weather_arrival=shipment.get("weather_condition_arrival") or "clear",
            carrier_on_time_rate=shipment.get("carrier_on_time_rate") or 0.9,
            route_type=shipment.get("route_type") or "highway",
        )
        return {
            "total_scheduled_min": prop_result.total_scheduled_min,
            "total_projected_min": prop_result.total_projected_min,
            "total_delay_min": prop_result.total_delay_min,
            "is_simulated": prop_result.is_simulated,
            "assumptions": prop_result.assumptions,
            "stages": [
                {
                    "stage": s.stage_name,
                    "scheduled_duration_min": s.scheduled_duration_min,
                    "additional_delay_min": s.additional_delay_min,
                    "cumulative_delay_min": s.cumulative_delay_min,
                }
                for s in prop_result.stages
            ],
        }

    # -------------------------------------------------------------- #
    #  Public API                                                      #
    # -------------------------------------------------------------- #

    def predict_single(
        self,
        shipment: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict ETA, delay risk, and explanation for a single shipment.

        Returns
        -------
        dict with keys:
            predicted_delay_min : float | None
            predicted_eta       : str  | None
            delay_risk          : str  | None   ("Delayed" / "On-Time")
            delay_probability   : float | None  (0-1)
            explanation         : dict
            propagation         : dict           (always populated)
            confidence_note     : str
        """
        result: dict[str, Any] = {
            "predicted_delay_min": None,
            "predicted_eta": None,
            "delay_risk": None,
            "delay_probability": None,
            "explanation": {},
            "propagation": {},
            "confidence_note": "",
        }

        X = self._build_features(shipment)

        # --- ETA regression ---
        if self.eta_model is not None and X is not None:
            try:
                pred_delay = float(self.eta_model.predict(X)[0])
                result["predicted_delay_min"] = round(pred_delay, 1)

                sched_arr = pd.to_datetime(
                    shipment.get("scheduled_arrival_datetime"),
                    utc=True,
                    errors="coerce",
                )
                if pd.notna(sched_arr):
                    result["predicted_eta"] = (
                        sched_arr + timedelta(minutes=pred_delay)
                    ).isoformat()
            except Exception as e:
                logger.error("ETA prediction failed: %s", e)

        # --- Delay classification ---
        if self.clf_model is not None and X is not None:
            try:
                pred_class = int(self.clf_model.predict(X)[0])
                result["delay_risk"] = (
                    "Delayed" if pred_class == 1 else "On-Time"
                )
                if hasattr(self.clf_model, "predict_proba"):
                    proba = self.clf_model.predict_proba(X)[0]
                    result["delay_probability"] = round(float(proba[1]), 3)
            except Exception as e:
                logger.error("Classification prediction failed: %s", e)

        # --- Explainability ---
        if self.eta_model is not None and X is not None:
            try:
                from src.explainability import explain_single_prediction

                feature_names = self.eta_meta.get(
                    "feature_columns", list(X.columns)
                )
                result["explanation"] = explain_single_prediction(
                    self.eta_model, X, feature_names,
                )
            except Exception as e:
                logger.warning("Explainability unavailable: %s", e)
                result["explanation"] = {
                    "note": f"Explainability unavailable: {e}"
                }
        else:
            result["explanation"] = {
                "note": (
                    "ML models not loaded — explainability unavailable. "
                    "Run 'python scripts/train_models.py' to train models."
                )
            }

        # --- Propagation (always runs — rule-based, no model dependency) ---
        result["propagation"] = self._run_propagation(shipment)

        # --- Confidence assessment ---
        if self.eta_model is None:
            result["confidence_note"] = (
                "ML models not loaded. "
                "Run 'python scripts/train_models.py' to enable predictions."
            )
        elif X is None:
            result["confidence_note"] = (
                "Feature engineering failed — check input data."
            )
        else:
            missing_count = int(X.isnull().sum().sum())
            if missing_count > 0:
                result["confidence_note"] = (
                    f"{missing_count} input feature(s) missing; "
                    "prediction confidence reduced."
                )
            else:
                result["confidence_note"] = "All input features present."

        return result

    def run_what_if(
        self,
        shipment: dict[str, Any],
        inject_stage: str,
        inject_delay_min: float,
    ) -> dict[str, Any]:
        """
        Run a what-if scenario: compare baseline propagation against a
        version with additional delay injected at a specific stage.
        """
        kwargs = dict(
            scheduled_duration_min=shipment.get("scheduled_duration_min") or 0,
            departure_delay_min=shipment.get("departure_delay_min") or 0,
            weather_departure=shipment.get("weather_condition_departure") or "clear",
            weather_arrival=shipment.get("weather_condition_arrival") or "clear",
            carrier_on_time_rate=shipment.get("carrier_on_time_rate") or 0.9,
            route_type=shipment.get("route_type") or "highway",
        )

        base = self.propagation.propagate(**kwargs)
        injected = self.propagation.propagate(
            **kwargs, custom_stage_delays={inject_stage: inject_delay_min},
        )

        return {
            "baseline_total_delay_min": base.total_delay_min,
            "injected_total_delay_min": injected.total_delay_min,
            "additional_impact_min": round(
                injected.total_delay_min - base.total_delay_min, 2,
            ),
            "baseline_stages": self.propagation.to_dataframe(base).to_dict(
                "records"
            ),
            "injected_stages": self.propagation.to_dataframe(injected).to_dict(
                "records"
            ),
            "is_simulated": True,
            "note": (
                f"What-if scenario: {inject_delay_min:.0f} min delay injected "
                f"at '{inject_stage}'. Stage timings are rule-based estimates."
            ),
        }
