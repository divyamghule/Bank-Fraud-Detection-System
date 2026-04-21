"""
Main Streamlit web UI for fraud detection system
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import (
    ANALYTICS_WINDOW_DAYS,
    AUDIT_LOG_PATH,
    DATASET_PATH,
    INDIAN_CITIES,
    PAYMENT_TYPES,
    VERIFICATION_TIMEOUT_SECONDS,
)
from fraud_engine import FraudDetectionEngine
from face_detection import verify_face_from_uploaded_file


# Page config
st.set_page_config(
    page_title="Bank Fraud Detection System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #ff6b6b;
        color: white;
    }
    .risk-medium {
        background-color: #ffd93d;
        color: black;
    }
    .risk-low {
        background-color: #51cf66;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "engine" not in st.session_state:
    if not DATASET_PATH.exists():
        st.error("Dataset not found! Run `python src/generate_dataset.py` first.")
        st.stop()
    st.session_state.engine = FraudDetectionEngine(DATASET_PATH)

if "verification_state" not in st.session_state:
    st.session_state.verification_state = None

if "current_result" not in st.session_state:
    st.session_state.current_result = None


engine = st.session_state.engine

# Sidebar
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["New Transaction", "Client Analytics", "Audit Logs"],
    index=0,
)


# ============================================================================
# MODE 1: NEW TRANSACTION
# ============================================================================
if app_mode == "New Transaction":
    st.title("🏦 Bank Fraud Detection System")
    st.subheader("Analyze Transaction Risk")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        client_name = st.selectbox(
            "Select Client",
            engine.df["client_name"].unique(),
            index=0,
        )
        client_id = engine.df[engine.df["client_name"] == client_name]["client_id"].iloc[0]
        verified_locations = engine._get_known_locations(client_id)
        other_locations = [location for location in INDIAN_CITIES if location not in verified_locations]
        location_options = verified_locations + other_locations
        
        payment_type = st.selectbox(
            "Payment Type",
            PAYMENT_TYPES,
            index=0,
        )
    
    with col2:
        if verified_locations:
            st.caption(f"Verified locations for {client_name}: {', '.join(verified_locations)}")
        location_city = st.selectbox(
            "Location",
            location_options,
            index=0,
        )
        
        amount = st.number_input(
            "Amount (₹)",
            min_value=100,
            max_value=50000,
            value=5000,
            step=500,
        )
    
    # Analyze button
    if st.button("🔍 Analyze Transaction", key="analyze_btn"):
        st.session_state.verification_state = None
        
        # Create transaction object
        transaction = {
            "client_id": client_id,
            "client_name": client_name,
            "transaction_id": f"TXN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "transaction_date": datetime.now(),
            "payment_type": payment_type,
            "location_city": location_city,
            "amount": amount,
            "merchant_category": "General",
        }
        
        # Predict
        result = engine.predict(transaction)
        st.session_state.current_result = result
    
    # Display result
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.divider()
        st.subheader("📋 Risk Assessment")
        
        # Risk display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_level = result["decision"]
            risk_color = {
                "BLOCK": "#ff6b6b",
                "VERIFY": "#ffd93d",
                "ALLOW": "#51cf66",
            }[risk_level]
            
            st.markdown(
                f"""
                <div style="
                    padding: 20px;
                    border-radius: 10px;
                    background-color: {risk_color};
                    color: white;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                ">
                Risk Level: {risk_level}
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            st.metric("Total Risk Score", f"{result['total_risk']:.2%}")
        
        with col3:
            st.metric("Rule Score", f"{result['rule_score']:.2%}")
        
        # Details
        st.subheader("Transaction Details")
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            st.write(f"**Client:** {result['client_id']}")
            st.write(f"**Amount:** ₹{result['amount']}")
            st.write(f"**Payment Type:** {result['payment_type']}")
        
        with details_col2:
            st.write(f"**Location:** {result['location']}")
            if result.get("known_locations"):
                st.write(f"**Known Locations:** {', '.join(result['known_locations'])}")
            if result.get("location_anomaly"):
                st.error("❌ Location not verified: transaction is outside client's most used locations")
            st.write(f"**ML Score:** {result['ml_score']:.2%}")
            st.write(f"**Time:** {result['timestamp']}")
        
        st.divider()
        
        # Decision-based handling
        if result["decision"] in ["VERIFY", "BLOCK"]:
            if result["decision"] == "BLOCK":
                st.error("❌ High Fraud Risk Detected")
                st.write("This transaction looks fraudulent. Live face verification is required before final decision.")
            else:
                st.warning("⚠️ Verification Required")
                st.write("Please verify with a live selfie to proceed with the transaction.")

            # Start verification button
            if st.button("📷 Start Face Verification", key="start_verification"):
                st.session_state.verification_started = True
            
            # Show camera only after button is clicked
            if st.session_state.get("verification_started", False):
                st.info("Capture your selfie - Make sure your face is clearly visible")
                captured_photo = st.camera_input("Take a Photo", key="camera_verify")

                if captured_photo is not None:
                    verification_result = verify_face_from_uploaded_file(captured_photo)
                    st.session_state.verification_state = verification_result
                    st.session_state.verification_started = False
        
        else:  # ALLOW
            st.success("✅ Transaction Approved")
            st.write("Your transaction has been approved successfully.")
    
    # Show verification result
    if st.session_state.verification_state:
        verification = st.session_state.verification_state
        st.divider()
        st.subheader("Verification Result")

        if verification.get("frame") is not None:
            st.image(verification["frame"], channels="BGR", caption="Detected Face Preview")
        
        if verification["verified"]:
            st.success("✅ Selfie Verification Successful!")
            st.write(
                f"Face detected: {verification['face_detected']}\n"
                f"Liveness confirmed: {verification['liveness_detected']}\n\n"
                f"**Status:** Transaction Approved ✅"
            )
        else:
            st.error("❌ Verification Failed")
            st.write(
                f"{verification['message']}\n\n"
                f"**Status:** Transaction Blocked ❌\n"
                f"Please try again or contact support."
            )
            
            if st.button("🔄 Retry Verification", key="retry_verify"):
                st.session_state.verification_state = None
                st.rerun()
    
    # 30-day analytics
    if st.session_state.current_result:
        st.divider()
        st.subheader("📈 30-Day Analytics")
        st.caption(f"Current month: {datetime.now().strftime('%B %Y')}")
        
        analytics = engine.get_client_analytics(client_id, days=ANALYTICS_WINDOW_DAYS)
        
        if "error" not in analytics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", analytics["total_transactions"])
            with col2:
                st.metric("Fraud Count", analytics["fraud_count"])
            with col3:
                st.metric("Fraud Rate", f"{analytics['fraud_rate']:.1%}")
            with col4:
                st.metric("Avg Amount", f"₹{analytics['avg_amount']:.0f}")
            
            # Line chart: Amount trend
            trans_df = pd.DataFrame(analytics["transactions"])
            trans_df["transaction_date"] = pd.to_datetime(trans_df["transaction_date"])
            trans_df = trans_df.sort_values("transaction_date")
            trans_df["date_label"] = trans_df["transaction_date"].dt.strftime("%d %b")
            trans_df["transaction_month"] = trans_df["transaction_date"].dt.strftime("%b %Y")
            
            fig_line = go.Figure()
            
            # Historical transactions
            fig_line.add_trace(go.Scatter(
                x=trans_df["transaction_date"],
                y=trans_df["amount"],
                mode='lines+markers',
                name='Historical Amount',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
            ))
            
            # Average line
            avg_amount = trans_df["amount"].mean()
            fig_line.add_hline(
                y=avg_amount,
                line_dash="dash",
                annotation_text="Average",
                annotation_position="right",
                line_color="green",
            )
            
            # Current transaction (if from same window)
            if st.session_state.current_result:
                current_date = datetime.now()
                fig_line.add_scatter(
                    x=[current_date],
                    y=[st.session_state.current_result["amount"]],
                    mode='markers',
                    name='Current Transaction',
                    marker=dict(size=12, color='red'),
                )
            
            fig_line.update_layout(
                title="Amount Trend (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="Amount (₹)",
                hovermode='x unified',
                height=400,
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Pie chart: Current transaction fraud percentage
            current_risk = float(st.session_state.current_result["total_risk"])
            labels = ["Fraud Risk %", "Safe %"]
            values = [
                round(current_risk * 100, 2),
                round((1 - current_risk) * 100, 2),
            ]
            colors = ["#ff6b6b", "#51cf66"]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textinfo='label+percent',
            )])
            
            fig_pie.update_layout(
                title="Current Transaction Fraud Probability",
                height=400,
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)


# ============================================================================
# MODE 2: CLIENT ANALYTICS
# ============================================================================
elif app_mode == "Client Analytics":
    st.title("📊 Client Analytics")
    
    client_name = st.selectbox(
        "Select Client",
        engine.df["client_name"].unique(),
    )
    
    client_id = engine.df[engine.df["client_name"] == client_name]["client_id"].iloc[0]
    analytics = engine.get_client_analytics(client_id, days=30)
    
    if "error" not in analytics:
        st.subheader(f"Analytics for {client_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", analytics["total_transactions"])
        with col2:
            st.metric("Fraud Count", analytics["fraud_count"])
        with col3:
            st.metric("Fraud Rate", f"{analytics['fraud_rate']:.1%}")
        with col4:
            st.metric("Average Amount", f"₹{analytics['avg_amount']:.0f}")
        
        st.divider()
        
        # Transactions table
        trans_df = pd.DataFrame(analytics["transactions"])
        trans_df["transaction_date"] = pd.to_datetime(trans_df["transaction_date"])
        trans_df = trans_df.sort_values("transaction_date", ascending=False)
        trans_df["date_label"] = trans_df["transaction_date"].dt.strftime("%d %b %Y, %I:%M %p")
        trans_df["transaction_month"] = trans_df["transaction_date"].dt.strftime("%b %Y")
        
        st.subheader("Recent Transactions")
        st.dataframe(
            trans_df[["transaction_id", "date_label", "transaction_month", "payment_type", "location_city", "amount", "is_fraud"]].rename(columns={"date_label": "transaction_date"}),
            use_container_width=True,
            height=400,
        )


# ============================================================================
# MODE 3: AUDIT LOGS
# ============================================================================
elif app_mode == "Audit Logs":
    st.title("📋 Audit Logs")
    
    audit_log_path = AUDIT_LOG_PATH
    
    if audit_log_path.exists():
        audit_df = pd.read_csv(audit_log_path, engine="python", on_bad_lines="skip")
        
        st.subheader(f"Total Decisions Logged: {len(audit_df)}")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Blocked", len(audit_df[audit_df["decision"] == "BLOCK"]))
        with col2:
            st.metric("Verified", len(audit_df[audit_df["decision"] == "VERIFY"]))
        with col3:
            st.metric("Allowed", len(audit_df[audit_df["decision"] == "ALLOW"]))
        
        st.divider()
        
        # Audit table
        st.subheader("All Decisions")
        st.dataframe(audit_df, use_container_width=True, height=500)
    else:
        st.info("No audit logs yet. Analyze some transactions to see logs here.")
