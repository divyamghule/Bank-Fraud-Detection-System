# 🏦 Bank Fraud Detection System (AI Project)

Complete end-to-end fraud detection system for multi-type bank transactions with real-time risk assessment, selfie verification, and 7-day behavioral analytics.

---

## 📋 Features

- **Multi-Type Fraud Detection**: Card, UPI, NEFT, Wallet transactions
- **Hybrid Risk Scoring**: 40% rule-based + 60% ML-based (RandomForest)
- **Real-Time Decision**: Block / Verify / Allow
- **Selfie Verification**: Live webcam capture with face detection (no photo matching)
- **7-Day Analytics**: Amount trend (line chart) + fraud distribution (pie chart)
- **Audit Logging**: All decision logs stored and queryable
- **Client Analytics**: Detailed transaction history and fraud patterns per client
- **Web UI**: Streamlit-based interactive dashboard

---

## 🏗️ Project Structure

```
ai bank froad/
├── src/
│   ├── __init__.py
│   ├── config.py              # Constants & config
│   ├── generate_dataset.py    # Synthetic data generation
│   ├── fraud_engine.py        # Core detection engine
│   ├── face_detection.py      # Webcam + face detection
│   ├── app.py                 # Streamlit UI
│   └── predict_sample.py      # Setup helper
├── data/                      # Generated dataset (auto-created)
├── models/                    # Trained model (auto-created)
├── logs/                      # Audit logs (auto-created)
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── PROJECT_BLUEPRINT.md       # Detailed system design
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python src/predict_sample.py
```

This creates:
- `data/transactions.csv` — Synthetic dataset (8 clients, 25 txns each)
- Auto-trains `models/fraud_model.joblib`

### 3. Run Web UI

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`

### Supported Python Version

Use Python 3.11 or 3.12. The project dependencies, especially `scikit-learn==1.6.1`, are most reliable on those versions. Python 3.14 can trigger build errors on some Windows machines.

The launcher scripts now try to pick Python 3.12 first, then 3.11, and automatically create the virtual environment.

### One-Command Launchers

- Windows: `run.bat`
- macOS/Linux: `./run.sh`

Both scripts will:
- select a supported Python version
- create `.venv` if missing
- upgrade `pip`, `setuptools`, and `wheel`
- install dependencies
- generate the dataset if needed
- start Streamlit

---

## 📊 Using the System

### Screen 1: New Transaction Analysis
1. Select client, payment type, location, amount
2. Click "🔍 Analyze Transaction"
3. View risk assessment (BLOCK / VERIFY / ALLOW)
4. See 7-day analytics (line + pie charts)

### Screen 2: Risk Decision Handling
- **BLOCK** (risk ≥ 0.75): Auto-blocked, no verification needed
- **VERIFY** (0.50–0.74): Requires selfie verification
- **ALLOW** (< 0.50): Approved directly

### Screen 3: Selfie Verification
- Opens webcam modal
- Detects any human face (not matched to client photo)
- Checks for liveness (movement/blink)
- Success → transaction approved
- Failure → blocked + manual review

### Screen 4: Client Analytics
- View 7-day history: fraud rate, avg amount, payment patterns
- See all transactions with dates, amounts, types

### Screen 5: Audit Logs
- All decisions logged with timestamp
- Searchable by client, decision type, date

---

## 🔍 How Fraud Detection Works

### Feature Extraction (8 features)
1. **Amount Deviation**: Z-score from 7-day average
2. **Location Anomaly**: 0 if seen before, 1 if unusual
3. **Velocity (1h)**: Transaction count in last hour
4. **Velocity (24h)**: Transaction count in last 24h
5. **Payment Type Anomaly**: 0 if used before, 1 if new
6. **Time Anomaly**: 0 if 6am-11pm, 1 if outside
7. **Amount Percentile**: % of txns below current amount
8. **Consecutive Txns**: Count of txns within <1h

### Risk Calculation
```
rule_score = (sum of 5 rules) ∈ [0, 1]
ml_score = RandomForest.predict_proba() ∈ [0, 1]
total_risk = (rule_score * 0.4) + (ml_score * 0.6)

IF total_risk >= 0.75: BLOCK
ELSE IF total_risk >= 0.50: VERIFY
ELSE: ALLOW
```

### Rules (40% weight)
1. **Amount anomaly**: > 2.5x mean → +0.35; > ₹40k → +0.15
2. **Location anomaly**: New city → +0.30; None → +0.15
3. **Velocity**: ≥3 txns/hr → +0.25; ≥8 txns/day → +0.15
4. **Payment type**: New type → +0.20
5. **Time anomaly**: Outside 6am-11pm → +0.10

---

## 🤖 ML Model

**Algorithm**: Random Forest Classifier
- 100 trees, max depth 10
- Balanced class weights (handles fraud imbalance)
- Trained on 7-day behavioral history

**Dataset**:
- 200 transactions (8 clients × 25 txns)
- ~15% fraud rate
- Splits: 80% train, 20% valida (internally)

---

## 📸 Selfie Verification Details

### How It Works
1. **Webcam Capture**: HTML5 MediaDevices API + JavaScript
2. **Face Detection**: OpenCV Haar Cascade (in Python backend)
3. **Verification**: 
   - Captures frames for 30 seconds (configurable)
   - Needs ≥5 frames with face detected
   - Liveness: Detects movement between frames
4. **Decision**: 
   - ✅ Success → transaction approved
   - ❌ Fail → blocked + retry allowed

### Important Notes
- **No photo matching**: Generic face detection only
- **Any human face passes**: Don't use client's stored photo
- **Privacy**: Selfie frames not stored (audit logs only store metadata)

---

## 📈 7-Day Analytics

### Line Chart: Amount Trend
- Blue line: Historical transaction amounts
- Green dashed: 7-day average
- Red dot: Current transaction
- Orange band: ±2 std dev range

### Pie Chart: Fraud Distribution
- Green: Legitimate transactions (%)
- Red: Fraudulent transactions (%)
- Counts last 7 days only

---

## 🔧 Configuration

Edit [`src/config.py`](src/config.py) to customize:

```python
RISK_THRESHOLDS = {
    "block": 0.75,      # Change to adjust block threshold
    "verify": 0.50,     # Change to adjust verify threshold
}

RULE_WEIGHT = 0.4      # Increase to trust rules more
MODEL_WEIGHT = 0.6     # Increase to trust ML more

AMOUNT_DEVIATION_MULTIPLIER = 2.5  # How many times mean = anomaly
VERIFICATION_TIMEOUT_SECONDS = 30   # Selfie timeout
```

---

## 📊 Sample Dataset

Generated automatically with:
- **8 unique clients** (realistic names)
- **25 transactions per client** (last 30 days)
- **4 payment types**: Card, UPI, NEFT, Wallet
- **10 Indian cities**: Mumbai, Delhi, Bangalore, Pune, etc.
- **10 merchant categories**: Grocery, Travel, Entertainment, etc.
- **~15% fraud rate** (balanced)
- **Amount range**: ₹100–₹50,000

---

## 🧪 Testing

### Manual Test: New Transaction
1. Run `streamlit run src/app.py`
2. Select client "Rajesh Kumar"
3. Try different amounts:
   - ₹5,000: Likely ALLOW
   - ₹45,000: Likely VERIFY (high amount)
   - Unusual location: Likely VERIFY
4. Test selfie verification with your webcam

### Test Fraud Patterns
Open `data/transactions.csv` and look for:
- `is_fraud=1` entries to see fraud patterns
- High amounts + remote locations
- Rapid velocity (multiple txns in minutes)

---

## 🛠️ Troubleshooting

### "Dataset not found"
```bash
python src/predict_sample.py
```

### "Streamlit not found"
```bash
pip install streamlit plotly opencv-python
```

### Camera not detected
- Check camera permissions (Windows/Mac/Linux)
- Try `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`
- If False, camera unavailable

### Model files missing
```bash
# Re-train (will happen automatically on app start)
python src/generate_dataset.py
```

---

## 📋 API Endpoints (Optional: FastAPI alternative)

If you want REST API instead of web UI:

```bash
uvicorn src.app:app --reload  # Requires FastAPI refactor
```

POST `/predict`:
```json
{
  "client_id": "C001",
  "amount": 5000,
  "location_city": "Mumbai",
  "payment_type": "Card",
  "transaction_date": "2026-03-24T10:30:00"
}
```

Response:
```json
{
  "decision": "VERIFY",
  "total_risk": 0.62,
  "rule_score": 0.65,
  "ml_score": 0.60
}
```

---

## 📚 Project Details

For detailed system design, see [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md):
- Complete architecture diagram
- Decision trees & rule logic
- UI mockups
- Data flow diagram
- Implementation phases

---

## 🎓 Learning Outcomes

This project demonstrates:
1. **Data Generation**: Realistic synthetic fraud dataset
2. **Feature Engineering**: Behavioral feature extraction
3. **Hybrid ML**: Rules + ML for interpretability
4. **Web Development**: Streamlit interactive dashboards
5. **Computer Vision**: Face detection (biometric security)
6. **Audit & Compliance**: Complete decision logging
7. **Real-Time Systems**: Sub-second fraud decisions

---

## 🚀 Next Steps / Extensions

- [ ] Real bank transaction dataset integration
- [ ] Email/SMS alerts for blocked transactions
- [ ] Dashboard for fraud analysts
- [ ] A/B testing for threshold optimization
- [ ] Advanced liveness detection (3D, anti-spoofing)
- [ ] Explainable AI (SHAP values)
- [ ] Deployment (Docker, Kubernetes, AWS)
- [ ] Multi-language UI (English, Marathi, Hindi)

---

## 📞 Support

For issues or questions:
1. Check [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md) for design details
2. Review `src/config.py` for tuning parameters
3. Check Streamlit logs: look for error messages

---

**Built with ❤️ | Python + Streamlit + scikit-learn + OpenCV**
