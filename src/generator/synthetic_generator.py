"""
Synthetic UPI Transaction Generator for PayWatch.
Generates realistic UPI transactions with normal behavioral profiles,
subtle injected fraud patterns, and benign edge cases (festivals, large purchases).
"""

import argparse
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune",
    "Chennai", "Kolkata", "Ahmedabad", "Nashik", "Jaipur",
    "Lucknow", "Indore", "Chandigarh", "Surat", "Kochi"
]

DEVICES = ["Android", "iOS", "web"]
UPI_CHANNELS = ["app", "QR", "intent_link"]

MERCHANT_CATEGORIES = {
    "groceries": (50.0, 1500.0),
    "food": (80.0, 1200.0),
    "utilities": (200.0, 4500.0),
    "travel": (150.0, 8000.0),
    "shopping": (300.0, 12000.0),
    "entertainment": (100.0, 2500.0),
    "electronics": (1500.0, 60000.0),
    "P2P": (50.0, 25000.0),
    "gambling": (500.0, 50000.0),
}


@dataclass
class UserProfile:
    user_id: str
    home_city: str
    device: str
    base_mean_spend: float
    base_std_spend: float
    is_night_owl: bool
    frequent_recipients: List[str] = field(default_factory=list)
    secondary_devices: List[str] = field(default_factory=list)


class UPIDataGenerator:
    """
    Simulates UPI transactions according to PRD FR-1 requirements.
    """

    def __init__(
        self,
        num_transactions: int = 12000,
        fraud_rate: float = 0.035,
        num_users: int = 500,
        num_merchants: int = 100,
        start_date: str = "2026-08-01T00:00:00+05:30",
        duration_days: int = 60,
        seed: int = 42,
    ):
        self.num_transactions = num_transactions
        self.fraud_rate = fraud_rate
        self.num_users = num_users
        self.num_merchants = num_merchants
        self.start_date_str = start_date
        self.duration_days = duration_days
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.users: Dict[str, UserProfile] = {}
        self.merchants: List[str] = []
        self._init_entities()

    def _init_entities(self):
        # Create merchant IDs
        self.merchants = [f"M_{100 + i}" for i in range(self.num_merchants)]

        # Create user profiles
        for i in range(1, self.num_users + 1):
            uid = f"U_{1000 + i}"
            home_city = str(self.rng.choice(CITIES))
            main_device = str(self.rng.choice(["Android", "iOS"], p=[0.75, 0.25]))
            is_night_owl = bool(self.rng.random() < 0.08)  # 8% legitimate night owls

            # Log-normal distribution for typical spend baseline
            base_mean = float(np.clip(self.rng.lognormal(mean=6.2, sigma=0.7), 200.0, 15000.0))
            base_std = float(max(50.0, base_mean * float(self.rng.uniform(0.25, 0.6))))

            frequent_recipients = [
                f"U_{1000 + random.randint(1, self.num_users)}" for _ in range(random.randint(2, 5))
            ] + [random.choice(self.merchants) for _ in range(random.randint(2, 4))]

            self.users[uid] = UserProfile(
                user_id=uid,
                home_city=home_city,
                device=main_device,
                base_mean_spend=round(base_mean, 2),
                base_std_spend=round(base_std, 2),
                is_night_owl=is_night_owl,
                frequent_recipients=frequent_recipients,
                secondary_devices=[d for d in DEVICES if d != main_device],
            )

    def _generate_timestamp(self, start_dt: datetime, is_night_owl: bool, force_odd_hour: bool = False) -> datetime:
        day_offset = self.rng.integers(0, self.duration_days)
        base_day = start_dt + timedelta(days=int(day_offset))

        if force_odd_hour:
            # High risk hours: 01:00 AM - 04:59 AM
            hour = int(self.rng.integers(1, 5))
        elif is_night_owl:
            # Night owls active late evening into night
            hour = int(self.rng.choice([21, 22, 23, 0, 1, 2, 10, 14, 18]))
        else:
            # Daytime curve (peak 10am-2pm, 6pm-10pm)
            hour_weights = [
                0.005, 0.005, 0.005, 0.005, 0.01, 0.02,  # 00-05
                0.04, 0.06, 0.07, 0.08, 0.09, 0.08,      # 06-11
                0.08, 0.07, 0.06, 0.05, 0.05, 0.06,      # 12-17
                0.08, 0.09, 0.08, 0.05, 0.03, 0.01       # 18-23
            ]
            hour_weights = np.array(hour_weights) / sum(hour_weights)
            hour = int(self.rng.choice(range(24), p=hour_weights))

        minute = int(self.rng.integers(0, 60))
        second = int(self.rng.integers(0, 60))
        return base_day.replace(hour=hour, minute=minute, second=second)

    def generate(self) -> pd.DataFrame:
        """
        Generates simulated dataset with both normal and fraudulent transactions.
        """
        # Parse start datetime (offset aware or naive string)
        clean_start_str = self.start_date_str.split("+")[0]
        start_dt = datetime.fromisoformat(clean_start_str)

        total_tx = self.num_transactions
        num_fraud = int(total_tx * self.fraud_rate)
        num_normal = total_tx - num_fraud

        records: List[Dict] = []
        user_ids = list(self.users.keys())

        # 1. Normal Transactions
        for i in range(num_normal):
            user = self.users[random.choice(user_ids)]
            ts = self._generate_timestamp(start_dt, user.is_night_owl)

            # Normal category & recipient
            is_p2p = self.rng.random() < 0.45
            if is_p2p:
                tx_type = "P2P"
                cat = "P2P"
                receiver = random.choice(user.frequent_recipients)
                channel = str(self.rng.choice(["app", "intent_link"], p=[0.85, 0.15]))
            else:
                tx_type = "P2M"
                cat = str(self.rng.choice([
                    "groceries", "food", "utilities", "travel", "shopping", "entertainment"
                ]))
                receiver = random.choice(self.merchants)
                channel = str(self.rng.choice(["QR", "app", "intent_link"], p=[0.60, 0.30, 0.10]))

            # Base amount with occasional noisy legitimate large purchases (festivals/electronics)
            is_festive_splurge = self.rng.random() < 0.015  # 1.5% legitimate splurge
            if is_festive_splurge:
                amount = float(self.rng.uniform(user.base_mean_spend * 2.5, user.base_mean_spend * 4.5))
                if not is_p2p:
                    cat = "electronics" if self.rng.random() < 0.5 else "shopping"
            else:
                amount = float(self.rng.normal(loc=user.base_mean_spend, scale=user.base_std_spend))
                amount = max(10.0, amount)

            # Location: 96% home city, 4% legitimate travel
            location = user.home_city if self.rng.random() < 0.96 else str(self.rng.choice(CITIES))
            # Device: 97% user device, 3% legitimate secondary device
            device = user.device if self.rng.random() < 0.97 else str(self.rng.choice(user.secondary_devices))

            records.append({
                "timestamp": ts,
                "sender_id": user.user_id,
                "receiver_id": receiver,
                "amount": round(amount, 2),
                "merchant_category": cat,
                "transaction_type": tx_type,
                "location": location,
                "device_type": device,
                "upi_channel": channel,
                "historical_spend_mean": user.base_mean_spend,
                "historical_spend_std": user.base_std_spend,
                "is_fraud": 0,
            })

        # 2. Injected Fraud Transactions
        fraud_archetypes = [
            "EXTREME_AMOUNT",
            "FREQUENCY_BURST",
            "ODD_HOURS_TAKEOVER",
            "REPEATED_DRAIN",
            "LOCATION_DEVICE_SWITCH",
        ]

        for i in range(num_fraud):
            user = self.users[random.choice(user_ids)]
            archetype = str(self.rng.choice(fraud_archetypes))

            if archetype == "EXTREME_AMOUNT":
                # 8x to 25x normal average spend
                amount = float(user.base_mean_spend * self.rng.uniform(7.0, 22.0))
                ts = self._generate_timestamp(start_dt, user.is_night_owl)
                receiver = f"U_{random.randint(9000, 9999)}"
                tx_type = "P2P"
                cat = "P2P"
                location = user.home_city
                device = user.device
                channel = "app"

            elif archetype == "FREQUENCY_BURST":
                # Rapid spike: high amount burst
                amount = float(user.base_mean_spend * self.rng.uniform(2.5, 6.0))
                ts = self._generate_timestamp(start_dt, user.is_night_owl)
                receiver = f"U_{random.randint(9000, 9999)}"
                tx_type = "P2P"
                cat = "gambling" if self.rng.random() < 0.4 else "P2P"
                location = user.home_city
                device = user.device
                channel = "app"

            elif archetype == "ODD_HOURS_TAKEOVER":
                # 1 AM - 4 AM, unusual device, new location, large amount
                amount = float(user.base_mean_spend * self.rng.uniform(4.0, 12.0))
                ts = self._generate_timestamp(start_dt, is_night_owl=False, force_odd_hour=True)
                receiver = f"U_{random.randint(9000, 9999)}"
                tx_type = "P2P"
                cat = "P2P"
                # Different city and unexpected device
                unseen_cities = [c for c in CITIES if c != user.home_city]
                location = str(self.rng.choice(unseen_cities))
                device = "web" if user.device != "web" else "Android"
                channel = "intent_link"

            elif archetype == "REPEATED_DRAIN":
                # Rapid transfers to dedicated drain recipient
                amount = float(user.base_mean_spend * self.rng.uniform(3.0, 8.0))
                ts = self._generate_timestamp(start_dt, user.is_night_owl)
                receiver = f"U_DRAIN_{random.randint(10, 30)}"
                tx_type = "P2P"
                cat = "P2P"
                location = user.home_city
                device = user.device
                channel = "app"

            else:  # LOCATION_DEVICE_SWITCH
                amount = float(user.base_mean_spend * self.rng.uniform(3.0, 7.5))
                ts = self._generate_timestamp(start_dt, user.is_night_owl)
                receiver = f"U_{random.randint(9000, 9999)}"
                tx_type = "P2P" if self.rng.random() < 0.7 else "P2M"
                cat = "gambling" if tx_type == "P2M" else "P2P"
                unseen_cities = [c for c in CITIES if c != user.home_city]
                location = str(self.rng.choice(unseen_cities))
                device = "web"
                channel = "intent_link"

            records.append({
                "timestamp": ts,
                "sender_id": user.user_id,
                "receiver_id": receiver,
                "amount": round(max(500.0, amount), 2),
                "merchant_category": cat,
                "transaction_type": tx_type,
                "location": location,
                "device_type": device,
                "upi_channel": channel,
                "historical_spend_mean": user.base_mean_spend,
                "historical_spend_std": user.base_std_spend,
                "is_fraud": 1,
            })

        df = pd.DataFrame(records)

        # Sort strictly chronologically to preserve real-time time-series sequence
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Inject some rapid burst duplicates in timestamps for burst-fraud cases
        fraud_indices = df[df["is_fraud"] == 1].index.tolist()
        for idx in fraud_indices[: min(50, len(fraud_indices))]:
            # Burst: ensure next 1-2 transactions for this user happen within 1-3 minutes
            if idx + 1 < len(df):
                df.at[idx + 1, "timestamp"] = df.at[idx, "timestamp"] + timedelta(seconds=random.randint(30, 150))
                df.at[idx + 1, "sender_id"] = df.at[idx, "sender_id"]

        # Re-sort chronologically after burst adjustments
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Generate unique transaction IDs: TXN_000001, TXN_000002...
        df["transaction_id"] = [f"TXN_{i+1:06d}" for i in range(len(df))]

        # Derived time fields per PRD FR-1
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Format timestamp to ISO 8601 with IST offset (+05:30)
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")

        # Organize column order per PRD FR-1
        cols_order = [
            "transaction_id",
            "timestamp",
            "sender_id",
            "receiver_id",
            "amount",
            "merchant_category",
            "transaction_type",
            "location",
            "device_type",
            "upi_channel",
            "hour_of_day",
            "day_of_week",
            "historical_spend_mean",
            "historical_spend_std",
            "is_fraud",
        ]
        return df[cols_order]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic UPI transaction dataset.")
    parser.add_argument("--n", type=int, default=12000, help="Number of transactions to generate (default: 12000)")
    parser.add_argument("--fraud-rate", type=float, default=0.035, help="Fraud rate proportion (default: 0.035)")
    parser.add_argument("--output", type=str, default="data/raw_transactions.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} transactions (fraud_rate={args.fraud_rate}, seed={args.seed})...")
    generator = UPIDataGenerator(
        num_transactions=args.n,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )
    df = generator.generate()
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} transactions successfully.")
    print(f"Fraud count: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Saved dataset to: {out_path}")


if __name__ == "__main__":
    main()
