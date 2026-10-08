"""
Time Series Anomaly Detector for PayWatch (FR-6).
Maintains rolling per-user baseline stats (rolling mean, rolling std, burst velocity)
and flags temporal anomalies, bursts, and shifts in streaming or batch modes.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TimeSeriesAnomalyDetector:
    """
    Streaming time-series anomaly detector.
    Tracks sliding temporal windows per user to detect:
    - Rolling amount spikes (amount > rolling_mean + N * rolling_std)
    - High-frequency burst clusters (multiple transactions within minutes)
    - Off-hours burst activity (1 AM - 5 AM unusual burst)
    """

    def __init__(
        self,
        rolling_window_hours: int = 24,
        std_multiplier: float = 2.8,
        burst_interval_seconds: int = 300,  # 5 minutes
        burst_count_threshold: int = 3,
        off_hours: Optional[List[int]] = None,
    ):
        self.rolling_window_seconds = rolling_window_hours * 3600
        self.std_multiplier = std_multiplier
        self.burst_interval_seconds = burst_interval_seconds
        self.burst_count_threshold = burst_count_threshold
        self.off_hours = off_hours if off_hours is not None else [1, 2, 3, 4]

        # In-memory sliding windows: user_id -> deque of (timestamp, amount)
        self.user_windows: Dict[str, deque] = defaultdict(deque)

    def _prune_stale(self, user_id: str, current_timestamp: float):
        queue = self.user_windows[user_id]
        cutoff = current_timestamp - self.rolling_window_seconds
        while queue and queue[0][0] < cutoff:
            queue.popleft()

    def evaluate_transaction(
        self,
        sender_id: str,
        amount: float,
        timestamp: Union[datetime, str],
        is_new_device: float = 0.0,
        update_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates a single transaction against the user's rolling time-series window.
        """
        if isinstance(timestamp, str):
            clean_ts = timestamp.split("+")[0]
            dt = datetime.fromisoformat(clean_ts)
        else:
            dt = timestamp

        ts_epoch = dt.timestamp()
        self._prune_stale(sender_id, ts_epoch)

        window = self.user_windows[sender_id]
        reasons: List[str] = []
        severity = 0.0

        # 1. Rolling statistics on window
        window_amounts = [amt for _, amt in window]
        window_count = len(window_amounts)

        if window_count >= 3:
            rolling_mean = float(np.mean(window_amounts))
            rolling_std = float(max(10.0, np.std(window_amounts)))
            rolling_threshold = rolling_mean + (self.std_multiplier * rolling_std)

            if amount > rolling_threshold:
                reasons.append("ROLLING_AMOUNT_SPIKE")
                excess = (amount - rolling_threshold) / rolling_std
                severity = max(severity, min(1.0, 0.4 + 0.15 * excess))
        else:
            rolling_mean = float(amount)
            rolling_std = float(amount * 0.5)

        # 2. Burst cluster detection (transactions within short burst interval)
        burst_cutoff = ts_epoch - self.burst_interval_seconds
        recent_burst_txs = [amt for t, amt in window if t >= burst_cutoff]
        # Include current transaction in count
        burst_count = len(recent_burst_txs) + 1

        is_burst = burst_count >= self.burst_count_threshold
        if is_burst:
            reasons.append("TIME_SERIES_BURST")
            severity = max(severity, 0.65 + min(0.3, burst_count * 0.05))

        # 3. Off-hours burst anomaly (1 AM - 5 AM with bursts or new device)
        if dt.hour in self.off_hours:
            if is_burst or is_new_device > 0:
                reasons.append("OFF_HOURS_BURST")
                severity = max(severity, 0.75)

        is_flagged = len(reasons) > 0
        severity = float(round(min(1.0, severity), 4))

        if update_state:
            self.user_windows[sender_id].append((ts_epoch, amount))

        return {
            "is_flagged": is_flagged,
            "score": severity if is_flagged else 0.0,
            "reasons": sorted(list(set(reasons))),
            "rolling_stats": {
                "rolling_mean": round(rolling_mean, 2),
                "rolling_std": round(rolling_std, 2),
                "window_count": window_count,
                "burst_count": burst_count,
            },
        }

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chronologically evaluates time-series anomalies across a DataFrame.
        """
        # Ensure chronological order
        sorted_df = df.sort_values("timestamp")
        flags = []
        scores = []
        reasons_list = []

        for _, row in sorted_df.iterrows():
            sender = str(row["sender_id"])
            amt = float(row["amount"])
            ts = row["timestamp"]
            new_dev = float(row.get("is_new_device", 0.0))

            res = self.evaluate_transaction(sender, amt, ts, is_new_device=new_dev, update_state=True)
            flags.append(int(res["is_flagged"]))
            scores.append(res["score"])
            reasons_list.append(";".join(res["reasons"]))

        res_df = pd.DataFrame({
            "ts_flag": flags,
            "ts_score": scores,
            "ts_reasons": reasons_list,
        }, index=sorted_df.index)

        # Restore original index order
        return res_df.loc[df.index]
