import pandas as pd
import matplotlib.pyplot as plt

# Spatial Data
spatial_data = {
    "Logic": [74, 73, 72, 64, 62, 66, 61, 67, 69, 58, 61, 61],
    "Hall": [18, 14, 9, 10, 9, 8, 5, 7, 1, 6, 1, 3],
    "Math": [11, 7, 8, 7, 7, 13, 8, 5, 17, 8, 16, 9],
    "Temp": [28, 26, 25, 31, 27, 25, 22, 20, 44, 25, 33, 25],
    "Vis": [65, 67, 69, 56, 61, 58, 68, 68, 72, 51, 67, 63],
}

# Temporal Data
temporal_data = {
    "Logic": [80, 72.5, 72.5, 63.3, 60.8, 68.3, 69.2, 65.0, 66.7, 65.0, 64.2, 60.8],
    "Hall": [36.7, 30.0, 30.8, 29.2, 23.3, 23.3, 30.0, 15.0, 14.2, 19.2, 11.7, 33.3],
    "Math": [10.0, 10.8, 15.0, 17.5, 17.5, 15.8, 10.8, 13.3, 17.5, 16.7, 9.2, 10.8],
    "Temp": [64.2, 62.5, 61.7, 68.3, 56.7, 75.0, 63.3, 59.2, 64.2, 75.0, 68.3, 70.8],
    "Vis": [50.8, 45.0, 47.5, 44.2, 45.8, 45.8, 50.8, 43.3, 51.7, 41.7, 45.0, 41.7],
}

# Create DataFrames
df_spatial = pd.DataFrame(spatial_data)
df_temporal = pd.DataFrame(temporal_data)

# Calculate averages
spatial_avg = df_spatial.mean()
temporal_avg = df_temporal.mean()

# Compile averages into a summary DataFrame
averages_summary = pd.DataFrame(
    {
        "Error Category": spatial_avg.index,
        "Spatial Average": spatial_avg.values,
        "Temporal Average": temporal_avg.values,
    }
)

# Save to CSV
averages_summary.to_csv("error_category_averages.csv", index=False)

# Plotting pie charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Pie chart for Spatial
ax1.pie(
    spatial_avg,
    labels=spatial_avg.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=plt.cm.Pastel1.colors,
)
ax1.set_title("Average Error Distribution: Spatial", fontweight="bold")

# Pie chart for Temporal
ax2.pie(
    temporal_avg,
    labels=temporal_avg.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=plt.cm.Pastel2.colors,
)
ax2.set_title("Average Error Distribution: Temporal", fontweight="bold")

plt.tight_layout()
plt.savefig("error_distribution_pies.png")

print("Averages for Spatial:")
print(spatial_avg)
print("\nAverages for Temporal:")
print(temporal_avg)
