"""
Generate synthetic bank transaction dataset
"""

import csv
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

from config import (
    DATASET_PATH,
    INDIAN_CITIES,
    MERCHANT_CATEGORIES,
    PAYMENT_TYPES,
    RANDOM_SEED,
)


def generate_dataset(n_clients: int = 8, txns_per_client: int = 25) -> pd.DataFrame:
    """Generate synthetic transaction dataset with realistic fraud patterns."""
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    rows = []
    client_names = [
        "Rajesh Kumar",
        "Priya Sharma",
        "Amit Patel",
        "Neha Singh",
        "Vikram Desai",
        "Anjali Malhotra",
        "Suresh K.",
        "Divya Reddy",
    ]
    
    for client_idx, client_name in enumerate(client_names[:n_clients]):
        client_id = f"C{str(client_idx + 1).zfill(3)}"
        
        # Each client has 20-25 transactions over last 30 days
        for txn_idx in range(txns_per_client):
            days_back = random.randint(1, 30)
            hours = random.randint(6, 23)
            minutes = random.randint(0, 59)
            
            txn_date = datetime.now() - timedelta(days=days_back, hours=hours, minutes=minutes)
            txn_date_str = txn_date.strftime("%Y-%m-%d %H:%M")
            
            # Client preferences (consistent patterns)
            primary_city = random.choice(INDIAN_CITIES[:5])  # Most txns in top 5 cities
            payment_type = random.choice(PAYMENT_TYPES)
            merchant_category = random.choice(MERCHANT_CATEGORIES)
            
            # Amount: log-normal distribution (realistic spending)
            amount = int(np.random.lognormal(mean=8.2, sigma=0.8)) 
            amount = max(100, min(amount, 50000))  # Clip to range
            
            # Fraud: ~15% fraud rate with patterns
            if random.random() < 0.15:
                is_fraud = 1
                # Fraud patterns: outlier amounts, unusual locations, rapid succession
                if random.random() < 0.5:
                    amount = random.randint(40000, 49999)  # High amount
                if random.random() < 0.4:
                    primary_city = random.choice(INDIAN_CITIES[5:])  # Unusual city
            else:
                is_fraud = 0
            
            txn_id = f"TXN{txn_date.strftime('%Y%m%d')}_{str(txn_idx).zfill(3)}"
            
            rows.append({
                "client_id": client_id,
                "client_name": client_name,
                "transaction_id": txn_id,
                "transaction_date": txn_date_str,
                "payment_type": payment_type,
                "location_city": primary_city,
                "amount": amount,
                "merchant_category": merchant_category,
                "is_fraud": is_fraud,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset created: {DATASET_PATH}")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    return df


if __name__ == "__main__":
    df = generate_dataset(n_clients=8, txns_per_client=25)
    print("\nDataset preview:")
    print(df.head(10))
