"""Deterministic sample datasets for tests and demos (stdlib only)."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "samples"


def write_csv(name: str, header: list[str], rows: list[list[object]]) -> None:
    path = OUT_DIR / name
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{path.relative_to(REPO_ROOT)}: {len(rows)} rows")


def binary_classification(rng: random.Random, n: int = 400) -> None:
    regions = ["North", "South", "East", "West"]
    channels = ["organic", "paid", "referral"]
    rows: list[list[object]] = []
    for _ in range(n):
        age = rng.randint(18, 70)
        income = round(max(15000.0, rng.gauss(60000, 15000)), 2)
        tenure = rng.randint(1, 72)
        region = rng.choice(regions)
        channel = rng.choices(channels, weights=[0.5, 0.3, 0.2])[0]
        logit = -1.5 + 0.02 * (age - 40) - 0.03 * tenure + (0.6 if channel == "paid" else 0.0)
        churned = 1 if rng.random() < 1 / (1 + math.exp(-logit)) else 0
        rows.append([age, income, tenure, region, channel, churned])
    write_csv(
        "binary_classification.csv",
        ["age", "income", "tenure_months", "region", "channel", "churned"],
        rows,
    )


def regression(rng: random.Random, n: int = 400) -> None:
    neighborhoods = {"A": 80000.0, "B": 40000.0, "C": 0.0, "D": -25000.0}
    rows: list[list[object]] = []
    for _ in range(n):
        sqft = round(rng.uniform(600, 3500), 1)
        bedrooms = rng.randint(1, 5)
        age_years = rng.randint(0, 100)
        neighborhood = rng.choices(list(neighborhoods), weights=[0.25, 0.3, 0.3, 0.15])[0]
        price = (
            50000
            + 180 * sqft
            + 15000 * bedrooms
            - 800 * age_years
            + neighborhoods[neighborhood]
            + rng.gauss(0, 25000)
        )
        rows.append([sqft, bedrooms, age_years, neighborhood, round(price, 2)])
    write_csv(
        "regression.csv",
        ["sqft", "bedrooms", "age_years", "neighborhood", "price"],
        rows,
    )


def clustering(rng: random.Random, n_per_cluster: int = 120) -> None:
    centers = [(0.0, 0.0), (6.0, 1.0), (2.0, 5.0)]
    rows: list[list[object]] = []
    for cluster_id, (mu_x, mu_y) in enumerate(centers):
        for _ in range(n_per_cluster):
            rows.append(
                [
                    round(rng.gauss(mu_x, 1.0), 4),
                    round(rng.gauss(mu_y, 1.0), 4),
                    cluster_id,
                ]
            )
    rng.shuffle(rows)
    write_csv("clustering.csv", ["x", "y", "true_cluster"], rows)


def messy_classification(rng: random.Random, n: int = 180) -> None:
    header = ["customer_id", "revenue", "region", "signup_days", "churned"]
    rows: list[list[object]] = []
    for index in range(n):
        revenue: object = round(max(0.0, rng.gauss(500, 200)), 2)
        draw = rng.random()
        if draw < 0.08:
            revenue = ""
        elif draw < 0.12:
            revenue = "N/A"
        region = rng.choices(
            ["North", "South", "East", "West", "north", "N"],
            weights=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05],
        )[0]
        churned = 1 if rng.random() < 0.15 else 0
        rows.append([f"C{index:04d}", revenue, region, rng.randint(1, 3650), churned])
    rows.extend(rows[:15])
    rng.shuffle(rows)
    write_csv("messy_classification.csv", header, rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    binary_classification(rng)
    regression(rng)
    clustering(rng)
    messy_classification(rng)


if __name__ == "__main__":
    main()
