"""
User History Tracker for Point-in-Time Feature Engineering.
Maintains stateful per-user behavioral baselines (spend history, velocity,
device/location history, recipient repetition) without look-ahead leakage.
Shared between batch training and real-time API scoring.
"""

from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class UserHistoryTracker:
    """
    Tracks historical activity for each user to compute behavioral features point-in-time.
    Supports streaming updates for incoming transactions.
    """

    def __init__(
        self,
        velocity_window_seconds: int = 3600,  # 1 hour
        recipient_window_seconds: int = 86400, # 24 hours
        global_mean_amount: float = 1200.0,
        global_std_amount: float = 1500.0,
        min_history_count: int = 5,
    ):
        self.velocity_window_seconds = velocity_window_seconds
        self.recipient_window_seconds = recipient_window_seconds
        self.global_mean_amount = global_mean_amount
        self.global_std_amount = global_std_amount
        self.min_history_count = min_history_count

        # In-memory user state
        self.user_amounts: Dict[str, List[float]] = defaultdict(list)
        self.user_timestamps: Dict[str, deque] = defaultdict(deque)
        self.user_recipients: Dict[str, deque] = defaultdict(deque) # deque of (timestamp, recipient_id)
        self.user_known_devices: Dict[str, Set[str]] = defaultdict(set)
        self.user_known_locations: Dict[str, Set[str]] = defaultdict(set)
        self.user_last_tx_time: Dict[str, datetime] = {}

    def get_user_stats(self, user_id: str) -> Tuple[float, float, int]:
        """
        Returns (mean_amount, std_amount, transaction_count).
        Falls back to global population stats if user is new or has insufficient history.
        """
        amounts = self.user_amounts.get(user_id, [])
        count = len(amounts)
        if count >= self.min_history_count:
            mean = float(np.mean(amounts))
            std = float(np.std(amounts))
            if std < 1e-4:
                std = self.global_std_amount
            return mean, std, count
        elif count > 0:
            # Partial history blended with global baseline
            mean = float(np.mean(amounts))
            return mean, self.global_std_amount, count
        else:
            # Complete cold-start
            return self.global_mean_amount, self.global_std_amount, 0

    def compute_features(
        self,
        sender_id: str,
        receiver_id: str,
        amount: float,
        timestamp: datetime,
        location: str,
        device_type: str,
        update_state: bool = True,
    ) -> Dict[str, float]:
        """
        Computes point-in-time features for a transaction BEFORE incorporating this transaction,
        ensuring strict no-leakage behavior.
        If update_state is True, updates the user's historical state with this transaction.
        """
        # 1. User spend stats prior to this transaction
        mean_amount, std_amount, tx_count = self.get_user_stats(sender_id)

        # Deviation features
        amount_ratio_to_mean = amount / max(1.0, mean_amount)
        amount_zscore = (amount - mean_amount) / max(1.0, std_amount)

        # 2. Time since last transaction
        last_time = self.user_last_tx_time.get(sender_id)
        if last_time is not None:
            time_since_last_sec = max(0.0, (timestamp - last_time).total_seconds())
        else:
            time_since_last_sec = 86400.0  # Default 24 hours for cold start

        # 3. Transaction velocity in the last 1 hour
        user_ts_queue = self.user_timestamps[sender_id]
        cutoff_velocity = timestamp.timestamp() - self.velocity_window_seconds
        # Clean up stale timestamps older than 1 hour
        while user_ts_queue and user_ts_queue[0] < cutoff_velocity:
            user_ts_queue.popleft()
        tx_count_last_1h = len(user_ts_queue)

        # 4. Recipient repetition count in the last 24 hours
        rec_queue = self.user_recipients[sender_id]
        cutoff_recipient = timestamp.timestamp() - self.recipient_window_seconds
        while rec_queue and rec_queue[0][0] < cutoff_recipient:
            rec_queue.popleft()
        recipient_tx_count_24h = sum(1 for _, r in rec_queue if r == receiver_id)

        # 5. Device and Location novelty
        known_devices = self.user_known_devices[sender_id]
        is_new_device = 1.0 if (known_devices and device_type not in known_devices) else 0.0

        known_locations = self.user_known_locations[sender_id]
        is_new_location = 1.0 if (known_locations and location not in known_locations) else 0.0

        # Derived calendar features
        hour_of_day = float(timestamp.hour)
        day_of_week = float(timestamp.weekday())
        is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0

        features = {
            "amount": float(amount),
            "amount_ratio_to_mean": float(round(amount_ratio_to_mean, 4)),
            "amount_zscore": float(round(amount_zscore, 4)),
            "user_mean_amount": float(round(mean_amount, 2)),
            "user_std_amount": float(round(std_amount, 2)),
            "user_tx_count_history": float(tx_count),
            "time_since_last_tx_sec": float(round(time_since_last_sec, 2)),
            "tx_count_last_1h": float(tx_count_last_1h),
            "recipient_tx_count_24h": float(recipient_tx_count_24h),
            "is_new_device": float(is_new_device),
            "is_new_location": float(is_new_location),
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        }

        # Update state point-in-time
        if update_state:
            self.user_amounts[sender_id].append(amount)
            self.user_timestamps[sender_id].append(timestamp.timestamp())
            self.user_recipients[sender_id].append((timestamp.timestamp(), receiver_id))
            self.user_known_devices[sender_id].add(device_type)
            self.user_known_locations[sender_id].add(location)
            self.user_last_tx_time[sender_id] = timestamp

        return features
