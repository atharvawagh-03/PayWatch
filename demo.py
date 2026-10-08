"""
PayWatch Live Interactive Demonstration Script.
Executes real-world UPI payment scenarios against the live running FastAPI service:
1. Normal Everyday Payment (Low Risk -> Allowed)
2. Unusual Splurge (Medium Risk -> Review)
3. Midnight Account Takeover / Drain Attack (Critical Risk -> Escalate)
4. Rapid Velocity Burst Attack (High Risk -> Escalate)
5. Analyst Review & Disposition Workflow
"""

import json
import time
import httpx

API_BASE_URL = "http://localhost:8000"


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)


def run_demo():
    print_separator("PayWatch Real-Time UPI Fraud Detection - Live Demo")
    print(f"Connecting to live backend at: {API_BASE_URL}...")

    with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
        # Check service health
        try:
            h = client.get("/health").json()
            print(f"Server Status: {h['status'].upper()} (Engine Version: {h['version']})")
        except Exception as e:
            print(f"Error connecting to API: {e}. Please ensure uvicorn is running.")
            return

        # Scenario 1: Normal Payment
        print_separator("Scenario 1: Legitimate Everyday UPI Payment")
        print("Profile: User U_1042 pays Rs 280 for groceries at 2:30 PM in Mumbai on Android.")
        txn_1 = {
            "transaction_id": "TXN_DEMO_NORMAL_01",
            "timestamp": "2026-10-08T14:30:00+05:30",
            "sender_id": "U_1042",
            "receiver_id": "M_105",
            "amount": 280.0,
            "merchant_category": "groceries",
            "transaction_type": "P2M",
            "location": "Mumbai",
            "device_type": "Android",
            "upi_channel": "QR"
        }
        r1 = client.post("/score", json=txn_1).json()
        print(f"\n[Scoring Result]:")
        print(f" -> Fraud Score : {r1['fraud_score']} / 100")
        print(f" -> Risk Level  : {r1['risk_level'].upper()}")
        print(f" -> Decision    : {r1['decision'].upper()}")
        print(f" -> Reason Codes: {r1['reasons'] if r1['reasons'] else 'None (Clean Transaction)'}")

        # Scenario 2: Moderate deviation
        print_separator("Scenario 2: Borderline / Elevated Spend")
        print("Profile: User U_1042 sends Rs 3,500 to a friend via P2P (above usual average).")
        txn_2 = {
            "transaction_id": "TXN_DEMO_REVIEW_02",
            "timestamp": "2026-10-08T15:45:00+05:30",
            "sender_id": "U_1042",
            "receiver_id": "U_1088",
            "amount": 3500.0,
            "merchant_category": "P2P",
            "transaction_type": "P2P",
            "location": "Mumbai",
            "device_type": "Android",
            "upi_channel": "app"
        }
        r2 = client.post("/score", json=txn_2).json()
        print(f"\n[Scoring Result]:")
        print(f" -> Fraud Score : {r2['fraud_score']} / 100")
        print(f" -> Risk Level  : {r2['risk_level'].upper()}")
        print(f" -> Decision    : {r2['decision'].upper()}")
        print(f" -> Reason Codes: {r2['reasons']}")

        # Scenario 3: Midnight Account Takeover
        print_separator("Scenario 3: Midnight Account Takeover & Drain Attack")
        print("Profile: 2:14 AM transfer of Rs 75,000 from unrecognized Web browser in Nashik.")
        txn_3 = {
            "transaction_id": "TXN_DEMO_FRAUD_03",
            "timestamp": "2026-10-08T02:14:00+05:30",
            "sender_id": "U_1042",
            "receiver_id": "U_DRAIN_99",
            "amount": 75000.0,
            "merchant_category": "P2P",
            "transaction_type": "P2P",
            "location": "Nashik",
            "device_type": "web",
            "upi_channel": "intent_link"
        }
        r3 = client.post("/score", json=txn_3).json()
        print(f"\n[Scoring Result]:")
        print(f" -> Fraud Score : {r3['fraud_score']} / 100")
        print(f" -> Risk Level  : {r3['risk_level'].upper()} [CRITICAL ALERT]")
        print(f" -> Decision    : {r3['decision'].upper()}")
        print(f" -> Reason Codes: {', '.join(r3['reasons'])}")
        print(f"\n[Multi-Layer Breakdown]:")
        for detector, details in r3.get("detector_breakdown", {}).items():
            flag_str = "FLAGGED" if details.get("flagged") else "OK"
            print(f"   * {detector.upper():<16}: [{flag_str}] (Score: {details.get('score')})")

        # Scenario 4: Velocity Burst Attack
        print_separator("Scenario 4: High-Frequency Velocity Burst")
        print("Profile: Same user executes rapid successive transfers within 60 seconds.")
        for i in range(1, 4):
            burst_txn = {
                "transaction_id": f"TXN_DEMO_BURST_{i:02d}",
                "timestamp": f"2026-10-08T16:0{i}:00+05:30",
                "sender_id": "U_BURST_ATTACK",
                "receiver_id": "U_MULE_12",
                "amount": 15000.0,
                "merchant_category": "P2P",
                "transaction_type": "P2P",
                "location": "Pune",
                "device_type": "Android",
                "upi_channel": "app"
            }
            res_burst = client.post("/score", json=burst_txn).json()
            print(f"   Transfer {i}: Rs 15,000 -> Score: {res_burst['fraud_score']}, Risk: {res_burst['risk_level']}, Reasons: {res_burst['reasons']}")

        # Scenario 5: Analyst Review Workflow
        print_separator("Scenario 5: Fraud Analyst Review Queue & Disposition")
        print("Fetching flagged transactions from SQLite store...")
        flagged_list = client.get("/flagged?risk_level=critical&limit=3").json()
        print(f"Total Critical Flagged in Database: {flagged_list['total']}")

        target_tx_id = "TXN_DEMO_FRAUD_03"
        print(f"\nAnalyst inspects transaction: {target_tx_id}")
        print("Analyst confirms fraud disposition with notes...")

        review_payload = {
            "status": "confirmed_fraud",
            "notes": "Cardholder contacted. Phishing credential harvest leading to web takeover."
        }
        reviewed = client.patch(f"/flagged/{target_tx_id}/review", json=review_payload).json()
        print(f"\n[Updated Record Status]:")
        print(f" -> Review Status: {reviewed['review_status'].upper()}")
        print(f" -> Notes        : {reviewed['reviewer_notes']}")
        print(f" -> Reviewed At  : {reviewed['reviewed_at']}")

        print_separator("Demo Finished Successfully")
        print("Explore all endpoints interactively in your browser at: http://localhost:8000/docs\n")


if __name__ == "__main__":
    run_demo()
