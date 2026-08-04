"""
Synthetic Batch Data Generator
================================================================================
Generates `synthetic_batch_data.csv`: a realistic synthetic dataset simulating
Fresh Fruit Bunch (FFB) batches moving through a CPO mill's intake pipeline.

The target variable, `ffa_percentage`, is produced via a domain-informed
formula (not pure noise) so the downstream XGBoost model in Module 2 learns
genuine, explainable relationships:

    - Longer harvest_delay_hours -> more bruising / lipase (enzymatic) activity
      -> higher FFA
    - Higher storage_temp_c      -> accelerated hydrolysis of triglycerides
      -> higher FFA
    - Higher humidity_percent    -> promotes microbial + hydrolytic activity
      -> higher FFA
    - Higher ripeness_score      -> overripe/rotted fruit already carries
      elevated native FFA before it even reaches the mill

A mild heat x humidity interaction term is added since the two together
accelerate spoilage faster than either does alone.

Run directly to (re)generate the CSV:
    python generate_data.py
"""

import numpy as np
import pandas as pd

# Module 1 assigns one of these integer ripeness weights
RIPENESS_LABELS = {0: "Underripe", 1: "Ripe", 2: "Overripe", 3: "Rotted"}

# Native FFA contribution per ripeness category (percentage points)
RIPENESS_FFA_EFFECT = {0: -0.20, 1: 0.00, 2: 0.65, 3: 1.60}

# Realistic prior on how a harvested batch is distributed across ripeness
# categories -- most fruit arriving at the mill should be Ripe/Underripe
RIPENESS_DISTRIBUTION = {0: 0.30, 1: 0.45, 2: 0.18, 3: 0.07}

FFA_FLOOR = 0.15  # a batch can never plausibly read below this


def calculate_ffa(harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score, rng):
    """Domain-informed FFA formula with a touch of realistic lab/sensor noise.
    All inputs are numpy arrays of equal length; returns an array of ffa_percentage.
    """
    base_ffa = 0.55

    delay_effect = 0.026 * harvest_delay_hours
    temp_effect = 0.045 * np.maximum(0, storage_temp_c - 25)
    humidity_effect = 0.016 * np.maximum(0, humidity_percent - 60)
    ripeness_effect = np.vectorize(RIPENESS_FFA_EFFECT.get)(ripeness_score)

    # Heat + humidity together accelerate degradation faster than either alone
    interaction_effect = 0.0009 * np.maximum(0, storage_temp_c - 30) * np.maximum(0, humidity_percent - 70)

    noise = rng.normal(0, 0.18, size=len(harvest_delay_hours))

    ffa = (
        base_ffa
        + delay_effect
        + temp_effect
        + humidity_effect
        + ripeness_effect
        + interaction_effect
        + noise
    )
    return np.clip(ffa, FFA_FLOOR, None)


def generate_dataset(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic batch dataset with realistic feature distributions."""
    rng = np.random.default_rng(seed)

    harvest_delay_hours = rng.uniform(0, 72, n_rows)
    storage_temp_c = rng.uniform(25, 45, n_rows)
    humidity_percent = rng.uniform(60, 100, n_rows)
    ripeness_score = rng.choice(
        list(RIPENESS_DISTRIBUTION.keys()),
        size=n_rows,
        p=list(RIPENESS_DISTRIBUTION.values()),
    )

    ffa_percentage = calculate_ffa(
        harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score, rng
    )

    df = pd.DataFrame({
        "harvest_delay_hours": np.round(harvest_delay_hours, 2),
        "storage_temp_c": np.round(storage_temp_c, 2),
        "humidity_percent": np.round(humidity_percent, 2),
        "ripeness_score": ripeness_score.astype(int),
        "ffa_percentage": np.round(ffa_percentage, 3),
    })
    return df


def main():
    df = generate_dataset()
    df.to_csv("synthetic_batch_data.csv", index=False)
    print(f"Generated {len(df)} rows -> synthetic_batch_data.csv\n")
    print("Summary statistics:")
    print(df.describe().round(2))
    print("\nRipeness distribution:")
    print(df["ripeness_score"].map(RIPENESS_LABELS).value_counts())
    print("\nRows landing in each FFA risk band:")
    bands = pd.cut(df["ffa_percentage"], bins=[0, 2.5, 3.5, 100], labels=["Safe (<2.5%)", "Warning (2.5-3.5%)", "Critical (>3.5%)"])
    print(bands.value_counts().sort_index())


if __name__ == "__main__":
    main()
