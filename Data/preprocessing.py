
import pandas as pd
import os


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#=====================================================
# Create folder for processed data
#=====================================================

os.makedirs("processed", exist_ok=True)

#=====================================================
#  Load Dataset
#=====================================================

df = pd.read_csv("data.csv")

print("Dataset loaded successfully")
print("Shape of dataset:", df.shape)

#=====================================================
#  Remove unnecessary columns
#=====================================================

# Some datasets include an ID column which has no predictive value
if "id" in df.columns:
    df = df.drop(columns=["id"])
    print("ID column removed")

#=====================================================
#  Create Target Variable
#=====================================================

# In this dataset, Thallium value of 3 indicates normal heart condition.
# Any other value indicates possible disease.

df["target"] = (df["Thallium"] != 3).astype(int)

print("Target variable created")

#=====================================================
#  Drop Thallium from Features
#=====================================================

# Thallium directly encodes the target so it must be
# removed from features to prevent data leakage
df = df.drop(columns=["Thallium"])
print("Thallium column dropped (prevents data leakage)")

#=====================================================
#  Check Missing Values
#=====================================================

print("\nChecking missing values...")
print(df.isnull().sum())

#=====================================================
#  Separate Features and Target
#=====================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
print("Features:", list(X.columns))

#=====================================================
#  Train / Test Split
#=====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain/Test split completed")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

#=====================================================
#  Feature Scaling
#=====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed")

# Convert scaled arrays back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

#=====================================================
#  Save Processed Data as CSV
#=====================================================

X_train_scaled.to_csv("processed/X_train.csv", index=False)
X_test_scaled.to_csv("processed/X_test.csv", index=False)

y_train.to_csv("processed/y_train.csv", index=False)
y_test.to_csv("processed/y_test.csv", index=False)

print("\nProcessed files saved")

print("""
Saved files:
processed/X_train.csv
processed/X_test.csv
processed/y_train.csv
processed/y_test.csv
""")