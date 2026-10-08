"""
Integration and End-to-End API Tests for PayWatch FastAPI Service (FR-8)
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.database import Base, engine, SessionLocal
from src.api.models import FlaggedTransaction


@pytest.fixture(scope="module")
def client():
    # Setup test database tables
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_score_normal_transaction(client):
    tx = {
        "transaction_id": "TXN_API_NORMAL_01",
        "timestamp": "2026-08-01T12:00:00+05:30",
        "sender_id": "U_1001",
        "receiver_id": "U_1002",
        "amount": 250.0,
        "merchant_category": "groceries",
        "transaction_type": "P2M",
        "location": "Mumbai",
        "device_type": "Android",
        "upi_channel": "QR",
    }
    response = client.post("/score", json=tx)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TXN_API_NORMAL_01"
    assert data["risk_level"] == "low"
    assert data["decision"] == "allow"
    assert data["fraud_score"] <= 35


def test_score_flagged_fraud_transaction(client):
    fraud_tx = {
        "transaction_id": "TXN_API_FRAUD_SCORE_01",
        "timestamp": "2026-08-01T02:14:00+05:30",
        "sender_id": "U_ATTACK",
        "receiver_id": "U_DRAIN_99",
        "amount": 88000.0,
        "merchant_category": "P2P",
        "transaction_type": "P2P",
        "location": "Nashik",
        "device_type": "web",
        "upi_channel": "intent_link",
    }
    response = client.post("/score", json=fraud_tx)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TXN_API_FRAUD_SCORE_01"
    assert data["risk_level"] in ["high", "critical"]
    assert data["decision"] in ["escalate", "escalate_with_priority"]
    assert data["fraud_score"] >= 70
    assert len(data["reasons"]) > 0

    # Verify transaction was persisted into flagged store
    flagged_resp = client.get(f"/flagged/{fraud_tx['transaction_id']}")
    assert flagged_resp.status_code == 200
    detail = flagged_resp.json()
    assert detail["transaction_id"] == "TXN_API_FRAUD_SCORE_01"
    assert detail["review_status"] == "pending"


def test_score_batch_endpoint(client):
    batch = {
        "transactions": [
            {
                "transaction_id": "TXN_BATCH_01",
                "timestamp": "2026-08-01T10:00:00+05:30",
                "sender_id": "U_1001",
                "receiver_id": "M_101",
                "amount": 150.0,
                "merchant_category": "food",
                "transaction_type": "P2M",
                "location": "Mumbai",
                "device_type": "Android",
                "upi_channel": "QR",
            },
            {
                "transaction_id": "TXN_BATCH_02",
                "timestamp": "2026-08-01T03:00:00+05:30",
                "sender_id": "U_1002",
                "receiver_id": "U_DRAIN_88",
                "amount": 95000.0,
                "merchant_category": "P2P",
                "transaction_type": "P2P",
                "location": "Jaipur",
                "device_type": "web",
                "upi_channel": "intent_link",
            },
        ]
    }
    response = client.post("/score/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2


def test_list_flagged_with_filters(client):
    response = client.get("/flagged?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)


def test_analyst_review_workflow(client):
    tx_id = "TXN_API_FRAUD_REVIEW_TARGET"
    # First score it so it enters flagged store
    fraud_tx = {
        "transaction_id": tx_id,
        "timestamp": "2026-08-01T02:30:00+05:30",
        "sender_id": "U_ATTACK_2",
        "receiver_id": "U_DRAIN_99",
        "amount": 75000.0,
        "merchant_category": "P2P",
        "transaction_type": "P2P",
        "location": "Kolkata",
        "device_type": "web",
        "upi_channel": "intent_link",
    }
    score_resp = client.post("/score", json=fraud_tx)
    assert score_resp.status_code == 200

    # Review with valid status
    review_payload = {
        "status": "confirmed_fraud",
        "notes": "Confirmed unauthorized device and location login burst."
    }
    patch_resp = client.patch(f"/flagged/{tx_id}/review", json=review_payload)
    assert patch_resp.status_code == 200
    updated = patch_resp.json()
    assert updated["review_status"] == "confirmed_fraud"
    assert updated["reviewer_notes"] == review_payload["notes"]
    assert updated["reviewed_at"] is not None

    # Invalid status should return 422
    bad_resp = client.patch(f"/flagged/{tx_id}/review", json={"status": "invalid_disposition"})
    assert bad_resp.status_code == 422
