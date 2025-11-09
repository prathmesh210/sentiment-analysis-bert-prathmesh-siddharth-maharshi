from src.data_extraction import load_data

df = load_data("data/sample_data.csv")

print("✅ Data loaded successfully!")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("Shape:", df.shape)
print("Any NaNs?", df.isna().sum().sum())
