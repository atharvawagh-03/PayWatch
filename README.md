# PayWatch
Real-time UPI fraud detection using IQR, Isolation Forest and time series anomaly detection, served through a FastAPI backend.

PayWatch is a layered fraud detection system for UPI transactions. It generates synthetic transaction data, engineers behavioral features, and combines three detection methods (IQR, Isolation Forest and time series analysis) into one explainable risk score. A FastAPI backend scores new transactions in real time and stores flagged ones for review.

Key features

10,000+ synthetic UPI transactions with injected fraud patterns
Behavioral features such as amount deviation, velocity, recipient repetition, and device and location change
Three-layer detection with risk levels and reason codes
Real-time scoring API with a flagged-transaction store
