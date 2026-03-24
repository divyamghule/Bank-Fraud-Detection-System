"""
Fraud detection engine: rules + ML model
"""

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    AMOUNT_DEVIATION_MULTIPLIER,
    AMOUNT_HIGH_THRESHOLD,
    ANALYTICS_WINDOW_DAYS,
    AUDIT_LOG_PATH,
    FEATURE_COLUMNS,
    HIGH_RISK_DISTANCE_KM,
    MODEL_WEIGHT,
    MODELS_DIR,
    RANDOM_SEED,
    RULE_WEIGHT,
    RISK_THRESHOLDS,
    VELOCITY_HIGH_24H,
    VELOCITY_RAPID_1H,
)


class FraudDetectionEngine:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.df["transaction_date"] = pd.to_datetime(self.df["transaction_date"])
        
        self.model = None
        self.scaler = StandardScaler()
        
        # Train model on historical data
        self._train_model()
    
    def _train_model(self):
        """Train RandomForest on the dataset."""
        # Feature engineering on historical data
        X, y = self._extract_features_for_training()
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Save model
        joblib.dump(
            {"model": self.model, "scaler": self.scaler},
            MODELS_DIR / "fraud_model.joblib"
        )
        print("Model trained and saved.")
    
    def _extract_features_for_training(self):
        """Extract features for all historical transactions."""
        X_list = []
        y_list = []
        
        for _, row in self.df.iterrows():
            features = self._calculate_features(row, self.df)
            if features is not None:
                X_list.append(features)
                y_list.append(row["is_fraud"])
        
        return np.array(X_list), np.array(y_list)
    
    def _get_client_history(self, client_id: str, days: int = ANALYTICS_WINDOW_DAYS) -> pd.DataFrame:
        """Get client's transaction history for last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        history = self.df[
            (self.df["client_id"] == client_id) &
            (self.df["transaction_date"] >= cutoff_date)
        ].copy()
        return history
    
    def _calculate_features(self, current_txn: pd.Series, historical_df: pd.DataFrame) -> np.ndarray | None:
        """Calculate feature vector for a transaction."""
        
        client_id = current_txn["client_id"]
        current_amount = current_txn["amount"]
        current_location = current_txn["location_city"]
        current_payment_type = current_txn["payment_type"]
        current_time = current_txn["transaction_date"]
        
        # Get client's 7-day history (excluding current transaction)
        history = self._get_client_history(client_id, days=ANALYTICS_WINDOW_DAYS)
        history = history[history["transaction_id"] != current_txn.get("transaction_id", "")]
        
        if len(history) == 0:
            return None  # Skip if no history
        
        # Feature 1: Amount deviation
        mean_amount = history["amount"].mean()
        std_amount = history["amount"].std()
        amount_deviation = (
            (current_amount - mean_amount) / max(std_amount, 1.0)
            if std_amount > 0 else 0
        )
        
        # Feature 2: Location anomaly (binary: 0 if seen before, 1 if new)
        location_anomaly = 0 if current_location in history["location_city"].values else 1
        
        # Feature 3: Velocity (transactions in last 1 hour)
        one_hour_ago = current_time - timedelta(hours=1)
        velocity_1h = len(
            history[history["transaction_date"] >= one_hour_ago]
        )
        
        # Feature 4: Velocity (transactions in last 24 hours)
        one_day_ago = current_time - timedelta(days=1)
        velocity_24h = len(
            history[history["transaction_date"] >= one_day_ago]
        )
        
        # Feature 5: Payment type anomaly
        payment_type_anomaly = 0 if current_payment_type in history["payment_type"].values else 1
        
        # Feature 6: Time anomaly (outside 6am–11pm)
        hour = current_time.hour
        time_anomaly = 0 if 6 <= hour < 23 else 1
        
        # Feature 7: Amount percentile
        amount_percentile = (history["amount"] < current_amount).sum() / len(history)
        
        # Feature 8: Consecutive transactions
        history_sorted = history.sort_values("transaction_date")
        if len(history_sorted) > 0:
            time_diffs = history_sorted["transaction_date"].diff().dt.total_seconds() / 3600  # hours
            consecutive_txns = (time_diffs < 1).sum()  # txns within 1 hour
        else:
            consecutive_txns = 0
        
        features = np.array([
            amount_deviation,
            location_anomaly,
            velocity_1h,
            velocity_24h,
            payment_type_anomaly,
            time_anomaly,
            amount_percentile,
            consecutive_txns,
        ])
        
        return features
    
    def _calculate_rules_score(self, current_txn: pd.Series, history: pd.DataFrame) -> float:
        """Calculate rule-based risk score (0–1)."""
        
        score = 0.0
        current_amount = current_txn["amount"]
        current_location = current_txn["location_city"]
        current_time = current_txn["transaction_date"]
        
        # Rule 1: Amount anomaly
        if len(history) > 0:
            mean_amount = history["amount"].mean()
            if current_amount > mean_amount * AMOUNT_DEVIATION_MULTIPLIER:
                score += 0.35
            elif current_amount > AMOUNT_HIGH_THRESHOLD:
                score += 0.15
        
        # Rule 2: Location anomaly
        if len(history) > 0 and current_location not in history["location_city"].values:
            score += 0.30
        else:
            score += 0.15 if len(history) == 0 else 0
        
        # Rule 3: Velocity check (rapid transactions)
        one_hour_ago = current_time - timedelta(hours=1)
        velocity_1h = len(history[history["transaction_date"] >= one_hour_ago])
        if velocity_1h >= VELOCITY_RAPID_1H:
            score += 0.25
        
        one_day_ago = current_time - timedelta(days=1)
        velocity_24h = len(history[history["transaction_date"] >= one_day_ago])
        if velocity_24h >= VELOCITY_HIGH_24H:
            score += 0.15
        
        # Rule 4: Payment type anomaly
        if len(history) > 0 and current_txn["payment_type"] not in history["payment_type"].values:
            score += 0.20
        
        # Rule 5: Time anomaly
        hour = current_time.hour
        if hour < 6 or hour >= 23:
            score += 0.10
        
        return min(score, 1.0)  # Cap at 1.0

    def _apply_amount_risk_policy(
        self,
        total_risk: float,
        current_amount: float,
        history: pd.DataFrame,
        client_id: str,
    ) -> float:
        """Apply client-specific amount policy on top of hybrid score.

        Policy:
        - Each client has a "high payment baseline" from historical data.
        - Risk starts increasing after baseline.
        - If amount >= 2x baseline => at least 95% risk.
        - If amount > 45000 => at least 95% risk.
        """
        adjusted_risk = total_risk

        if current_amount > 45000:
            return max(adjusted_risk, 0.95)

        client_full_history = self.df[self.df["client_id"] == client_id]
        if len(client_full_history) == 0 and len(history) > 0:
            client_full_history = history.copy()

        if len(client_full_history) == 0:
            return min(adjusted_risk, 1.0)

        # Baseline from mostly legitimate behavior
        if "is_fraud" in client_full_history.columns:
            legit_history = client_full_history[client_full_history["is_fraud"] == 0]
            baseline_source = legit_history if len(legit_history) > 0 else client_full_history
        else:
            baseline_source = client_full_history

        # High payment baseline = max(90th percentile, 1.4x average)
        avg_amount = float(baseline_source["amount"].mean())
        p90_amount = float(np.percentile(baseline_source["amount"], 90))
        high_payment_baseline = max(p90_amount, 1.4 * avg_amount)

        if high_payment_baseline <= 0:
            return min(adjusted_risk, 1.0)

        double_baseline = float(np.ceil(2 * high_payment_baseline))

        # Double or above => very high risk
        if current_amount >= double_baseline:
            adjusted_risk = max(adjusted_risk, 0.95)
            return min(adjusted_risk, 1.0)

        # Between baseline and double => ramp risk upward
        if current_amount > high_payment_baseline:
            span = max(double_baseline - high_payment_baseline, 1.0)
            progress = (current_amount - high_payment_baseline) / span
            progress = min(max(progress, 0.0), 1.0)

            # Starts around medium-high and increases to near-very-high
            ramp_risk = 0.55 + (0.35 * progress)
            adjusted_risk = max(adjusted_risk, ramp_risk)

        return min(adjusted_risk, 1.0)
    
    def predict(self, current_txn: dict) -> dict:
        """Predict fraud for a new transaction."""
        
        # Convert to pandas Series
        txn_series = pd.Series(current_txn)
        txn_series["transaction_date"] = pd.to_datetime(txn_series["transaction_date"])
        
        # Get client history
        history = self._get_client_history(txn_series["client_id"], days=ANALYTICS_WINDOW_DAYS)
        client_full_history = self.df[self.df["client_id"] == txn_series["client_id"]]
        
        # Calculate rule-based score
        rule_score = self._calculate_rules_score(txn_series, history)
        
        # Calculate ML score
        features = self._calculate_features(txn_series, self.df)
        if features is None:
            ml_score = 0.5  # Default for new clients
        else:
            features_scaled = self.scaler.transform([features])
            ml_prob = self.model.predict_proba(features_scaled)[0, 1]
            ml_score = ml_prob
        
        # Combine scores
        total_risk = (rule_score * RULE_WEIGHT) + (ml_score * MODEL_WEIGHT)
        total_risk = self._apply_amount_risk_policy(
            total_risk=float(total_risk),
            current_amount=float(txn_series["amount"]),
            history=history,
            client_id=str(txn_series["client_id"]),
        )
        
        # Determine decision
        if total_risk >= RISK_THRESHOLDS["block"]:
            decision = "BLOCK"
        elif total_risk >= RISK_THRESHOLDS["verify"]:
            decision = "VERIFY"
        else:
            decision = "ALLOW"
        
        result = {
            "client_id": txn_series["client_id"],
            "transaction_id": txn_series.get("transaction_id", "NEW_TXN"),
            "amount": current_txn["amount"],
            "location": current_txn["location_city"],
            "payment_type": current_txn["payment_type"],
            "avg_amount_7d": round(float(history["amount"].mean()), 2) if len(history) > 0 else None,
            "avg_amount_client": round(float(client_full_history["amount"].mean()), 2) if len(client_full_history) > 0 else None,
            "high_payment_baseline": round(
                float(
                    max(
                        np.percentile(
                            (client_full_history[client_full_history["is_fraud"] == 0]["amount"]
                             if ("is_fraud" in client_full_history.columns and len(client_full_history[client_full_history["is_fraud"] == 0]) > 0)
                             else client_full_history["amount"]),
                            90,
                        ),
                        1.4 * (
                            (client_full_history[client_full_history["is_fraud"] == 0]["amount"].mean()
                             if ("is_fraud" in client_full_history.columns and len(client_full_history[client_full_history["is_fraud"] == 0]) > 0)
                             else client_full_history["amount"].mean())
                        ),
                    )
                )
            , 2) if len(client_full_history) > 0 else None,
            "rule_score": round(rule_score, 4),
            "ml_score": round(ml_score, 4),
            "total_risk": round(total_risk, 4),
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Log to audit
        self._log_audit(result)
        
        return result
    
    def _log_audit(self, result: dict):
        """Log transaction decision to audit log."""
        audit_df = pd.DataFrame([result])
        
        if AUDIT_LOG_PATH.exists():
            audit_df.to_csv(AUDIT_LOG_PATH, mode='a', header=False, index=False)
        else:
            audit_df.to_csv(AUDIT_LOG_PATH, index=False)
    
    def get_client_analytics(self, client_id: str, days: int = ANALYTICS_WINDOW_DAYS) -> dict:
        """Get analytics for client (last N days)."""
        
        history = self._get_client_history(client_id, days=days)
        
        if len(history) == 0:
            return {"error": "No transactions found"}
        
        return {
            "client_id": client_id,
            "total_transactions": len(history),
            "fraud_count": history["is_fraud"].sum(),
            "fraud_rate": float(history["is_fraud"].mean()),
            "avg_amount": float(history["amount"].mean()),
            "max_amount": float(history["amount"].max()),
            "min_amount": float(history["amount"].min()),
            "cities": history["location_city"].unique().tolist(),
            "payment_types": history["payment_type"].unique().tolist(),
            "transactions": history.to_dict('records'),
        }
