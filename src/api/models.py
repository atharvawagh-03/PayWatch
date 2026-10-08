"""
SQLAlchemy ORM models for flagged transactions and analyst review storage.
"""

from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from src.api.database import Base


class FlaggedTransaction(Base):
    __tablename__ = "flagged_transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(String(64), nullable=False)
    sender_id = Column(String(64), index=True, nullable=False)
    receiver_id = Column(String(64), nullable=False)
    amount = Column(Float, nullable=False)
    location = Column(String(64), nullable=False)
    device_type = Column(String(64), nullable=False)
    upi_channel = Column(String(64), nullable=False)
    merchant_category = Column(String(64), nullable=False)
    transaction_type = Column(String(64), nullable=False)

    fraud_score = Column(Integer, nullable=False)
    risk_level = Column(String(32), index=True, nullable=False)
    decision = Column(String(32), nullable=False)
    reasons = Column(Text, nullable=False)  # JSON or comma-separated list
    detector_breakdown = Column(Text, nullable=True)

    # Analyst Review Status: 'pending', 'confirmed_fraud', 'false_positive'
    review_status = Column(String(32), default="pending", index=True, nullable=False)
    reviewer_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
