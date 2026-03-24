"""
Config and constants for fraud detection system
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_PATH = DATA_DIR / "transactions.csv"
AUDIT_LOG_PATH = LOGS_DIR / "audit_log.csv"

# Feature columns
FEATURE_COLUMNS = [
    "amount_deviation",
    "location_anomaly",
    "velocity_1h",
    "velocity_24h",
    "payment_type_anomaly",
    "time_anomaly",
    "amount_percentile",
    "consecutive_txns",
]

# Risk thresholds
RISK_THRESHOLDS = {
    "block": 0.75,        # >= 0.75: BLOCK
    "verify": 0.50,       # 0.50–0.74: VERIFY
    "allow": 0.0,         # < 0.50: ALLOW
}

# Rule weights (40% rules, 60% ML)
RULE_WEIGHT = 0.4
MODEL_WEIGHT = 0.6

# Anomaly detection parameters
AMOUNT_DEVIATION_MULTIPLIER = 2.5
AMOUNT_HIGH_THRESHOLD = 40000
VELOCITY_RAPID_1H = 3
VELOCITY_HIGH_24H = 8
HIGH_RISK_DISTANCE_KM = 500

# Selfie verification
VERIFICATION_TIMEOUT_SECONDS = 30

# Cities
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
]

# Payment types
PAYMENT_TYPES = ["Card", "UPI", "NEFT", "Wallet"]

# Merchant categories
MERCHANT_CATEGORIES = [
    "Grocery", "Online_Shopping", "Hotel", "ATM", "Fuel",
    "Travel", "Entertainment", "Medical", "Utilities", "Food_Delivery",
]

# 30-day window for analytics
ANALYTICS_WINDOW_DAYS = 30

# Random seed for reproducibility
RANDOM_SEED = 42
