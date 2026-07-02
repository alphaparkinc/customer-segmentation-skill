# customer-segmentation-skill

> **GenPark AI Agent Skill** — Segment e-commerce customers into behavioral groups using K-means clustering on RFM features.

## Features

- K-means++ initialization for better cluster quality
- RFM feature normalization (z-score)
- Auto-labeling: High Value Champions, Loyal Regulars, At-Risk, Dormant
- Marketing strategy per segment
- Silhouette score for cluster quality measurement
- No external dependencies — pure Python implementation

## Quick Start

```python
from client import CustomerSegmentationClient

client = CustomerSegmentationClient(seed=42)
result = client.segment(
    customers=[
        {"customer_id": "C001", "recency_days": 5, "frequency": 20, "monetary_value": 1200},
        {"customer_id": "C002", "recency_days": 180, "frequency": 2, "monetary_value": 45},
    ],
    n_segments=4,
)
for seg in result["segments"]:
    print(f"{seg['label']}: {seg['customer_count']} customers")
    print(f"Strategy: {seg['strategy']}")
```

## Installation

```bash
python example_usage.py  # No external dependencies
```

---
Built by [GenPark](https://genpark.ai) | [alphaparkinc](https://github.com/alphaparkinc)
