"""
customer-segmentation-skill: Client SDK
Segment e-commerce customers into behavioral groups using K-means on RFM features.
"""

from __future__ import annotations
import math
import random
from typing import Optional

SEGMENT_PROFILES = {
    "high_value": {
        "label": "High Value Champions",
        "description": "Recent, frequent, high-spending customers.",
        "strategy": "Reward with loyalty programs, exclusive early access, and VIP perks.",
    },
    "loyal": {
        "label": "Loyal Regulars",
        "description": "Consistent buyers with moderate spend.",
        "strategy": "Upsell to premium products, cross-sell complementary categories.",
    },
    "at_risk": {
        "label": "At-Risk Customers",
        "description": "Previously active customers showing decline.",
        "strategy": "Winback campaign with personalized discount and product recommendation.",
    },
    "dormant": {
        "label": "Dormant / Lost",
        "description": "Inactive customers with low historical engagement.",
        "strategy": "Aggressive re-engagement or sunset from active marketing.",
    },
    "new": {
        "label": "New Customers",
        "description": "First-time or very recent buyers.",
        "strategy": "Welcome sequence, onboarding offers, second-purchase incentive.",
    },
}


class CustomerSegmentationClient:
    """
    SDK for behavioral customer segmentation using K-means clustering on RFM features.

    Implements K-means with:
      - Feature normalization (z-score)
      - K-means++ initialization
      - Convergence detection
      - Silhouette score estimation
      - Auto-labeling based on segment centroids
    """

    def __init__(self, seed: Optional[int] = None, max_iterations: int = 100):
        self.seed = seed
        self.max_iterations = max_iterations
        if seed is not None:
            random.seed(seed)

    def segment(
        self,
        customers: list[dict],
        n_segments: int = 4,
        features: Optional[list[str]] = None,
    ) -> dict:
        """
        Segment customers into behavioral groups.

        Args:
            customers:   List of dicts with at minimum: recency_days, frequency, monetary_value.
                         Also accepts: customer_id, email, name.
            n_segments:  Number of clusters (2-8).
            features:    Feature names to cluster on (default: recency_days, frequency, monetary_value).

        Returns:
            dict with: segments, customers (with labels), silhouette_score, n_segments
        """
        if not customers:
            return {"segments": [], "customers": [], "silhouette_score": 0, "n_segments": 0}

        n_segments = max(2, min(n_segments, min(8, len(customers))))
        features = features or ["recency_days", "frequency", "monetary_value"]

        # Extract and normalize feature matrix
        matrix, valid_customers = self._extract_features(customers, features)
        normalized, means, stds = self._normalize(matrix)

        # K-means clustering
        labels, centroids = self._kmeans(normalized, n_segments)

        # Assign labels to customers
        for i, cust in enumerate(valid_customers):
            cust["_cluster"] = int(labels[i])

        # Build segment profiles
        segments = self._build_segments(valid_customers, centroids, means, stds, features, n_segments)

        # Silhouette score
        sil_score = self._silhouette(normalized, labels, n_segments)

        # Map cluster IDs to human labels
        cluster_label_map = self._auto_label_clusters(segments)
        for cust in valid_customers:
            cust["segment"] = cluster_label_map.get(cust["_cluster"], f"Segment {cust['_cluster']}")
            del cust["_cluster"]

        return {
            "segments": segments,
            "customers": valid_customers,
            "silhouette_score": round(sil_score, 3),
            "n_segments": n_segments,
            "features_used": features,
        }

    def _extract_features(self, customers: list[dict], features: list[str]) -> tuple:
        valid = []
        matrix = []
        for c in customers:
            try:
                row = [float(c.get(f, 0)) for f in features]
                valid.append(dict(c))
                matrix.append(row)
            except (TypeError, ValueError):
                continue
        return matrix, valid

    def _normalize(self, matrix: list[list[float]]) -> tuple:
        if not matrix:
            return [], [], []
        n_features = len(matrix[0])
        means = [sum(row[j] for row in matrix) / len(matrix) for j in range(n_features)]
        stds = [
            math.sqrt(sum((row[j] - means[j]) ** 2 for row in matrix) / len(matrix)) or 1.0
            for j in range(n_features)
        ]
        normalized = [[(row[j] - means[j]) / stds[j] for j in range(n_features)] for row in matrix]
        return normalized, means, stds

    def _kmeans(self, data: list[list[float]], k: int) -> tuple:
        """K-means with k-means++ initialization."""
        # Init centroids with k-means++
        centroids = [list(data[random.randint(0, len(data) - 1)])]
        while len(centroids) < k:
            dists = [min(self._dist(x, c) for c in centroids) for x in data]
            total = sum(dists)
            r = random.uniform(0, total)
            cumsum = 0
            for i, d in enumerate(dists):
                cumsum += d
                if cumsum >= r:
                    centroids.append(list(data[i]))
                    break

        labels = [0] * len(data)
        for _ in range(self.max_iterations):
            # Assign
            new_labels = [min(range(k), key=lambda c: self._dist(x, centroids[c])) for x in data]
            if new_labels == labels:
                break
            labels = new_labels
            # Update centroids
            for c in range(k):
                cluster_pts = [data[i] for i, l in enumerate(labels) if l == c]
                if cluster_pts:
                    centroids[c] = [sum(pt[j] for pt in cluster_pts) / len(cluster_pts) for j in range(len(data[0]))]

        return labels, centroids

    @staticmethod
    def _dist(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _build_segments(self, customers, centroids, means, stds, features, n_segments) -> list[dict]:
        segments = []
        for c in range(n_segments):
            members = [cust for cust in customers if cust.get("_cluster") == c]
            if not members:
                continue
            # Denormalize centroid
            centroid_raw = {
                features[j]: round(centroids[c][j] * stds[j] + means[j], 2)
                for j in range(len(features))
            }
            segments.append({
                "cluster_id": c,
                "customer_count": len(members),
                "centroid": centroid_raw,
                "avg_recency_days": round(sum(m.get("recency_days", 0) for m in members) / len(members), 1),
                "avg_frequency": round(sum(m.get("frequency", 0) for m in members) / len(members), 1),
                "avg_monetary": round(sum(m.get("monetary_value", 0) for m in members) / len(members), 2),
                "label": f"Segment {c}",
                "strategy": "",
            })
        return segments

    def _silhouette(self, data: list[list[float]], labels: list[int], k: int) -> float:
        """Approximate silhouette score for quality measurement."""
        if len(set(labels)) < 2:
            return 0.0
        scores = []
        for i, point in enumerate(data[:min(len(data), 200)]):
            a = self._intra_dist(point, i, labels, data)
            b = self._min_inter_dist(point, i, labels, data, k)
            if max(a, b) > 0:
                scores.append((b - a) / max(a, b))
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _intra_dist(point, idx, labels, data) -> float:
        cluster = labels[idx]
        peers = [data[j] for j, l in enumerate(labels) if l == cluster and j != idx]
        if not peers:
            return 0.0
        return sum(math.sqrt(sum((a - b) ** 2 for a, b in zip(point, p))) for p in peers) / len(peers)

    @staticmethod
    def _min_inter_dist(point, idx, labels, data, k) -> float:
        my_cluster = labels[idx]
        min_dist = float("inf")
        for c in range(k):
            if c == my_cluster:
                continue
            peers = [data[j] for j, l in enumerate(labels) if l == c]
            if peers:
                avg = sum(math.sqrt(sum((a - b) ** 2 for a, b in zip(point, p))) for p in peers) / len(peers)
                min_dist = min(min_dist, avg)
        return min_dist if min_dist < float("inf") else 0.0

    @staticmethod
    def _auto_label_clusters(segments: list[dict]) -> dict:
        """Map cluster IDs to human-readable labels based on RFM centroids."""
        labeled = {}
        for seg in sorted(segments, key=lambda s: (s["avg_monetary"], -s["avg_recency_days"]), reverse=True):
            rank = len(labeled)
            if rank == 0:
                label = SEGMENT_PROFILES["high_value"]["label"]
                seg["strategy"] = SEGMENT_PROFILES["high_value"]["strategy"]
            elif rank == 1:
                label = SEGMENT_PROFILES["loyal"]["label"]
                seg["strategy"] = SEGMENT_PROFILES["loyal"]["strategy"]
            elif rank == len(segments) - 1:
                label = SEGMENT_PROFILES["dormant"]["label"]
                seg["strategy"] = SEGMENT_PROFILES["dormant"]["strategy"]
            else:
                label = SEGMENT_PROFILES["at_risk"]["label"]
                seg["strategy"] = SEGMENT_PROFILES["at_risk"]["strategy"]
            seg["label"] = label
            labeled[seg["cluster_id"]] = label
        return labeled
