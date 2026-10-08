"""
Pydantic Schemas for PayWatch FastAPI Service (FR-8).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_id: str = Field(..., json_schema_extra={"example": "TXN_000123"})
    timestamp: str = Field(..., json_schema_extra={"example": "2026-10-08T02:14:00+05:30"})
    sender_id: str = Field(..., json_schema_extra={"example": "U_1042"})
    receiver_id: str = Field(..., json_schema_extra={"example": "U_8891"})
    amount: float = Field(..., gt=0, json_schema_extra={"example": 48500.0})
    merchant_category: str = Field("P2P", json_schema_extra={"example": "P2P"})
    transaction_type: str = Field("P2P", json_schema_extra={"example": "P2P"})
    location: str = Field("Nashik", json_schema_extra={"example": "Nashik"})
    device_type: str = Field("Android", json_schema_extra={"example": "Android"})
    upi_channel: str = Field("app", json_schema_extra={"example": "app"})


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: int = Field(..., ge=0, le=100)
    risk_level: str = Field(..., json_schema_extra={"example": "high"})
    decision: str = Field(..., json_schema_extra={"example": "escalate"})
    reasons: List[str] = Field(default_factory=list)
    model_version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    detector_breakdown: Optional[Dict[str, Any]] = None


class BatchScoreRequest(BaseModel):
    transactions: List[TransactionInput]


class BatchScoreResponse(BaseModel):
    total_processed: int
    flagged_count: int
    results: List[ScoreResponse]


class FlaggedTransactionResponse(BaseModel):
    id: int
    transaction_id: str
    timestamp: str
    sender_id: str
    receiver_id: str
    amount: float
    location: str
    device_type: str
    upi_channel: str
    fraud_score: int
    risk_level: str
    decision: str
    reasons: List[str]
    review_status: str
    reviewer_notes: Optional[str] = None
    created_at: str
    reviewed_at: Optional[str] = None


class FlaggedListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[FlaggedTransactionResponse]


class ReviewRequest(BaseModel):
    status: str = Field(..., pattern="^(confirmed_fraud|false_positive)$", json_schema_extra={"example": "confirmed_fraud"})
    notes: Optional[str] = Field(None, json_schema_extra={"example": "Verified with cardholder. Account compromised."})


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: str
