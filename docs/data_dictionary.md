# PayWatch: Data Dictionary & Simulation Methodology

## 1. Overview
This dataset contains simulated UPI (Unified Payments Interface) transactions engineered to model normal human payment patterns, benign statistical outliers, and coordinated fraud vectors. The data is generated reproducibly via `src/generator/synthetic_generator.py`.

---

## 2. Dataset Schema

| Field Name | Data Type | Example Value | Description |
| :--- | :--- | :--- | :--- |
| `transaction_id` | String | `TXN_000123` | Unique identifier for each transaction. |
| `timestamp` | String (ISO 8601) | `2026-08-01T14:22:15+05:30` | Transaction creation timestamp with Indian Standard Time (+05:30) offset. |
| `sender_id` | String | `U_1042` | Pseudonymous identifier of the sending user. |
| `receiver_id` | String | `U_8891` or `M_105` | Pseudonymous recipient identifier (User or Merchant). |
| `amount` | Float | `4850.00` | Transaction value in Indian Rupees (INR). |
| `merchant_category`| Categorical | `groceries`, `P2P`, `gambling` | Business classification category of the payment. |
| `transaction_type` | Categorical | `P2P`, `P2M` | Payment rail type (Peer-to-Peer or Peer-to-Merchant). |
| `location` | String | `Mumbai`, `Bengaluru` | Originating city/metro area. |
| `device_type` | Categorical | `Android`, `iOS`, `web` | Operating system / client device used for the transaction. |
| `upi_channel` | Categorical | `app`, `QR`, `intent_link` | UPI payment mechanism channel. |
| `hour_of_day` | Integer (0-23) | `14` | Derived hour of the day in local time. |
| `day_of_week` | Integer (0-6) | `5` | Derived day of week (0 = Monday, 6 = Sunday). |
| `historical_spend_mean` | Float | `1450.25` | User's rolling baseline historical mean spend in INR. |
| `historical_spend_std` | Float | `420.10` | User's rolling baseline standard deviation of spend. |
| `is_fraud` | Integer (0, 1) | `0` (Normal) or `1` (Fraud) | **Ground-truth label**. Used strictly for model evaluation and test reporting; never fed as an input feature into anomaly detectors. |

---

## 3. Allowed Categorical Domains

- **`merchant_category`**: `groceries`, `food`, `utilities`, `travel`, `shopping`, `entertainment`, `electronics`, `P2P`, `gambling`.
- **`transaction_type`**: `P2P`, `P2M`.
- **`device_type`**: `Android`, `iOS`, `web`.
- **`upi_channel`**: `app`, `QR`, `intent_link`.
- **`location`**: Top Indian economic hubs including `Mumbai`, `Delhi`, `Bengaluru`, `Hyderabad`, `Pune`, `Chennai`, `Kolkata`, `Ahmedabad`, `Nashik`, `Jaipur`, `Lucknow`, `Indore`, `Chandigarh`, `Surat`, `Kochi`.

---

## 4. Behavior Simulation Architecture

### 4.1 Normal Behavior
- **Spend Baselines**: Each user has an individual log-normal spend distribution characterized by `historical_spend_mean` and `historical_spend_std`.
- **Device & Location Affinity**: 96%+ of transactions originate from the user's registered home city and primary smartphone.
- **Payee Locality**: Regular payments flow to a fixed pool of frequent contacts and favorite local merchants.
- **Benign Edge Cases (Noise)**:
  - High-ticket legitimate purchases (e.g., festival season gifts, electronics) occurring during business hours on recognized devices.
  - Legitimate night-shift workers with consistent off-peak activity profiles.

### 4.2 Injected Fraud Vectors (~3.5% default rate)
1. **Extreme Amount Spikes**: Rapid deviation where the transaction amount is 7x to 22x greater than the user's typical average.
2. **Velocity Bursts**: Multiple rapid transfers occurring within a window of a few minutes.
3. **Account Takeover (ATO)**: Transactions executed between 01:00 AM and 04:59 AM from an unfamiliar device (e.g., `web`) in an unexpected city with elevated amounts.
4. **Drain Transfers**: Fast-paced repeated transfers to dedicated mule or drain accounts (`U_DRAIN_*`).
5. **Channel / Location Hijack**: Sudden switch to collect requests (`intent_link` or `web`) from an unvisited geographic location.
