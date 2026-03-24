"""
Quick setup and run instructions
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import generate_dataset
from config import DATASET_PATH


def main():
    print("=" * 60)
    print("🏦 Bank Fraud Detection System - Setup")
    print("=" * 60)
    
    # Step 1: Generate dataset
    if not DATASET_PATH.exists():
        print("\n📊 Generating synthetic dataset...")
        df = generate_dataset(n_clients=8, txns_per_client=25)
        print(f"✅ Dataset created: {DATASET_PATH}")
        print(f"   Total transactions: {len(df)}")
        print(f"   Fraud rate: {df['is_fraud'].mean():.2%}")
    else:
        print(f"\n✅ Dataset already exists: {DATASET_PATH}")
    
    print("\n" + "=" * 60)
    print("🚀 Ready to Run!")
    print("=" * 60)
    print("\nTo start the web UI, run:")
    print("  streamlit run src/app.py")
    print("\nOr from the project root:")
    print("  streamlit run ./src/app.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
