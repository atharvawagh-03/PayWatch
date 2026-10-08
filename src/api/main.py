"""
PayWatch FastAPI Backend Application (FR-8).
Real-time scoring API, flagged transaction store, and analyst review workflows.
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.database import Base, engine, get_db
from src.api.models import FlaggedTransaction
from src.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    FlaggedListResponse,
    FlaggedTransactionResponse,
    HealthResponse,
    ReviewRequest,
    ScoreResponse,
    TransactionInput,
)
from src.config import get_config
from src.ensemble import EnsembleFraudEngine

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paywatch_api")

config = get_config()
ensemble_engine: Optional[EnsembleFraudEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ensemble_engine
    logger.info("Initializing SQLite database tables...")
    Base.metadata.create_all(bind=engine)

    logger.info("Loading pre-trained fraud detection ensemble artifacts...")
    try:
        ensemble_engine = EnsembleFraudEngine.load_from_artifacts()
        logger.info("EnsembleFraudEngine loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load saved artifacts ({e}), initializing fallback engine...")
        from src.detectors.iqr_detector import IQROutlierDetector
        from src.detectors.iso_forest_detector import IsoForestAnomalyDetector
        import pandas as pd

        # Quick initialization fallback if models not pre-built
        df = pd.read_csv("data/engineered_transactions.csv")
        iqr = IQROutlierDetector().fit(df)
        iso = IsoForestAnomalyDetector(n_estimators=30).fit(df[["amount", "amount_ratio_to_mean"]].values)
        ensemble_engine = EnsembleFraudEngine(iqr_detector=iqr, iso_detector=iso)

    yield
    logger.info("Shutting down PayWatch service...")


app = FastAPI(
    title=config.api.get("title", "PayWatch Real-time UPI Fraud Detection API"),
    version=config.version,
    description="Real-time UPI fraud detection using IQR, Isolation Forest, and Time Series Anomaly Detection.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _persist_flagged_if_needed(tx_dict: dict, score_result: dict, db: Session):
    """
    Persists transactions to SQLite if risk level is medium, high, or critical.
    """
    if score_result["risk_level"] in ["medium", "high", "critical"]:
        existing = db.query(FlaggedTransaction).filter(
            FlaggedTransaction.transaction_id == score_result["transaction_id"]
        ).first()

        if not existing:
            flagged_entry = FlaggedTransaction(
                transaction_id=score_result["transaction_id"],
                timestamp=tx_dict.get("timestamp", datetime.now(timezone.utc).isoformat()),
                sender_id=tx_dict.get("sender_id", "UNKNOWN"),
                receiver_id=tx_dict.get("receiver_id", "UNKNOWN"),
                amount=float(tx_dict.get("amount", 0.0)),
                location=tx_dict.get("location", "UNKNOWN"),
                device_type=tx_dict.get("device_type", "UNKNOWN"),
                upi_channel=tx_dict.get("upi_channel", "UNKNOWN"),
                merchant_category=tx_dict.get("merchant_category", "P2P"),
                transaction_type=tx_dict.get("transaction_type", "P2P"),
                fraud_score=score_result["fraud_score"],
                risk_level=score_result["risk_level"],
                decision=score_result["decision"],
                reasons=json.dumps(score_result["reasons"]),
                detector_breakdown=json.dumps(score_result.get("detector_breakdown", {})),
                review_status="pending",
                created_at=datetime.now(timezone.utc),
            )
            db.add(flagged_entry)
            db.commit()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="healthy",
        version=config.version,
        environment="production",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_transaction(payload: TransactionInput, db: Session = Depends(get_db)):
    """
    Scores an incoming transaction in real-time.
    Returns composite fraud score, risk level, action, and human-readable reason codes.
    Flagged transactions are automatically saved to the review store.
    """
    if ensemble_engine is None:
        raise HTTPException(status_code=500, detail="Fraud detection engine not initialized.")

    tx_dict = payload.model_dump()
    result = ensemble_engine.score_single(tx_dict, update_state=True)

    logger.info(
        f"Scored TX {payload.transaction_id}: Score={result['fraud_score']}, "
        f"Risk={result['risk_level']}, Decision={result['decision']}, Reasons={result['reasons']}"
    )

    _persist_flagged_if_needed(tx_dict, result, db)
    return ScoreResponse(**result)


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
def score_batch_transactions(payload: BatchScoreRequest, db: Session = Depends(get_db)):
    """
    Scores a batch of transactions and stores all flagged items.
    """
    if ensemble_engine is None:
        raise HTTPException(status_code=500, detail="Fraud detection engine not initialized.")

    results = []
    flagged_count = 0
    for tx in payload.transactions:
        tx_dict = tx.model_dump()
        res = ensemble_engine.score_single(tx_dict, update_state=True)
        _persist_flagged_if_needed(tx_dict, res, db)
        if res["risk_level"] in ["medium", "high", "critical"]:
            flagged_count += 1
        results.append(ScoreResponse(**res))

    return BatchScoreResponse(
        total_processed=len(payload.transactions),
        flagged_count=flagged_count,
        results=results,
    )


@app.get("/flagged", response_model=FlaggedListResponse, tags=["Analyst Review"])
def list_flagged_transactions(
    risk_level: Optional[str] = Query(None, description="Filter by risk level (medium, high, critical)"),
    sender_id: Optional[str] = Query(None, description="Filter by sender user ID"),
    review_status: Optional[str] = Query(None, description="Filter by review status (pending, confirmed_fraud, false_positive)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Retrieves paginated flagged transactions for fraud analyst inspection.
    """
    query = db.query(FlaggedTransaction)

    if risk_level:
        query = query.filter(FlaggedTransaction.risk_level == risk_level)
    if sender_id:
        query = query.filter(FlaggedTransaction.sender_id == sender_id)
    if review_status:
        query = query.filter(FlaggedTransaction.review_status == review_status)

    total = query.count()
    items = query.order_by(FlaggedTransaction.id.desc()).offset(offset).limit(limit).all()

    formatted_items = []
    for item in items:
        reasons_list = json.loads(item.reasons) if item.reasons else []
        formatted_items.append(
            FlaggedTransactionResponse(
                id=item.id,
                transaction_id=item.transaction_id,
                timestamp=item.timestamp,
                sender_id=item.sender_id,
                receiver_id=item.receiver_id,
                amount=item.amount,
                location=item.location,
                device_type=item.device_type,
                upi_channel=item.upi_channel,
                fraud_score=item.fraud_score,
                risk_level=item.risk_level,
                decision=item.decision,
                reasons=reasons_list,
                review_status=item.review_status,
                reviewer_notes=item.reviewer_notes,
                created_at=item.created_at.isoformat(),
                reviewed_at=item.reviewed_at.isoformat() if item.reviewed_at else None,
            )
        )

    return FlaggedListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=formatted_items,
    )


@app.get("/flagged/{transaction_id}", response_model=FlaggedTransactionResponse, tags=["Analyst Review"])
def get_flagged_transaction_detail(transaction_id: str, db: Session = Depends(get_db)):
    """
    Retrieves detailed breakdown of a specific flagged transaction.
    """
    item = db.query(FlaggedTransaction).filter(FlaggedTransaction.transaction_id == transaction_id).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Flagged transaction {transaction_id} not found.")

    reasons_list = json.loads(item.reasons) if item.reasons else []
    return FlaggedTransactionResponse(
        id=item.id,
        transaction_id=item.transaction_id,
        timestamp=item.timestamp,
        sender_id=item.sender_id,
        receiver_id=item.receiver_id,
        amount=item.amount,
        location=item.location,
        device_type=item.device_type,
        upi_channel=item.upi_channel,
        fraud_score=item.fraud_score,
        risk_level=item.risk_level,
        decision=item.decision,
        reasons=reasons_list,
        review_status=item.review_status,
        reviewer_notes=item.reviewer_notes,
        created_at=item.created_at.isoformat(),
        reviewed_at=item.reviewed_at.isoformat() if item.reviewed_at else None,
    )


@app.patch("/flagged/{transaction_id}/review", response_model=FlaggedTransactionResponse, tags=["Analyst Review"])
def review_flagged_transaction(
    transaction_id: str,
    payload: ReviewRequest,
    db: Session = Depends(get_db),
):
    """
    Allows a fraud analyst to submit a disposition: marks transaction as confirmed_fraud or false_positive.
    """
    item = db.query(FlaggedTransaction).filter(FlaggedTransaction.transaction_id == transaction_id).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Flagged transaction {transaction_id} not found.")

    item.review_status = payload.status
    if payload.notes:
        item.reviewer_notes = payload.notes
    item.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(item)

    logger.info(f"Analyst reviewed TX {transaction_id}: Outcome={item.review_status}, Notes={item.reviewer_notes}")

    reasons_list = json.loads(item.reasons) if item.reasons else []
    return FlaggedTransactionResponse(
        id=item.id,
        transaction_id=item.transaction_id,
        timestamp=item.timestamp,
        sender_id=item.sender_id,
        receiver_id=item.receiver_id,
        amount=item.amount,
        location=item.location,
        device_type=item.device_type,
        upi_channel=item.upi_channel,
        fraud_score=item.fraud_score,
        risk_level=item.risk_level,
        decision=item.decision,
        reasons=reasons_list,
        review_status=item.review_status,
        reviewer_notes=item.reviewer_notes,
        created_at=item.created_at.isoformat(),
        reviewed_at=item.reviewed_at.isoformat() if item.reviewed_at else None,
    )
