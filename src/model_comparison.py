
import os
import pandas as pd

# Create output directory
os.makedirs("results/metrics", exist_ok=True)

# Test-set results from your evaluation outputs
results = [
    {
        "Model": "Custom CNN",
        "Accuracy": 0.6789,
        "Precision": 0.5898,
        "Recall": 0.6789,
        "F1_Score": 0.5929,
        "Test_Loss": 0.7323
    },
    {
        "Model": "EfficientNetB0",
        "Accuracy": 0.8740,
        "Precision": 0.8738,
        "Recall": 0.8740,
        "F1_Score": 0.8697,
        "Test_Loss": 0.3383
    }
]

# Create DataFrame
df = pd.DataFrame(results)

# Print comparison
print("\nMODEL COMPARISON")
print("=" * 75)
print(df.to_string(index=False))

# Save CSV
output_path = "results/metrics/model_comparison.csv"
df.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")
