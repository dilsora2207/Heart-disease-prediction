import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# create folder for plots
os.makedirs("plots", exist_ok=True)

# load dataset
df = pd.read_csv("data.csv")

# remove id column
df = df.drop(columns=["id"])

# create target variable
df["target"] = (df["Thallium"] != 3).astype(int)

print("Dataset Shape:", df.shape)
print(df.head())

print(df.info())
print(df.describe())

print("Missing Values:")
print(df.isnull().sum())

# =========================
# Colors
# =========================

BLUE = "#4C9BD5"
RED = "#E45756"
ORANGE = "#F5A623"
GREEN = "#63B365"
PURPLE = "#9B59B6"

sns.set_style("whitegrid")




# =====================================================
#  Target Distribution
# =====================================================

plt.figure(figsize=(6,6))
counts = df['target'].value_counts()

plt.pie(
    counts,
    labels=["Normal (0)","Disease (1)"],
    colors=[BLUE,RED],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Target Variable Distribution")
plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.close()


# =====================================================
#  Age Distribution
# =====================================================

plt.figure(figsize=(6,4))
plt.hist(df[df.target==0]["Age"],bins=30,alpha=0.6,color=BLUE,label="Normal")
plt.hist(df[df.target==1]["Age"],bins=30,alpha=0.6,color=RED,label="Disease")

plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution by Class")
plt.legend()
plt.tight_layout()
plt.savefig("plots/age_distribution.png")
plt.close()

# =====================================================
#  Age vs Max HR
# =====================================================

sample = df.sample(4000)

plt.figure(figsize=(6,4))

plt.scatter(
    sample["Age"],
    sample["Max HR"],
    c=sample["target"].map({0:BLUE,1:BLUE}),
    alpha=0.5
)

plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.title("Age vs Max HR")
plt.tight_layout()
plt.savefig("plots/age_vs_maxhr.png")

# =====================================================
#  ST Depression vs Disease
# =====================================================

plt.figure(figsize=(6,4))
sns.boxplot(x='target',y='ST depression',data=df,palette=[BLUE,RED])
plt.title('ST Depression vs Disease')
plt.tight_layout()
plt.savefig('plots/st_depression_vs_disease.png')
plt.close()

# =====================================================
#  Cholesterol vs Disease
# =====================================================

plt.figure(figsize=(6,4))
sns.boxplot(x='target',y='Cholesterol',data=df,palette=[BLUE,RED])
plt.title('Cholesterol vs Disease')
plt.tight_layout()
plt.savefig('plots/cholesterol_vs_disease.png')
plt.close()

# =====================================================
#  Correlation Matrix
# =====================================================

plt.figure(figsize=(8,6))
corr = df.corr()

sns.heatmap(corr,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

# =====================================================
#  Chest Pain Type vs Disease
# =====================================================

cp_rate = df.groupby('Chest pain type')['target'].mean()*100

plt.figure(figsize=(6,4))
plt.bar(
    cp_rate.index.astype(str),
    cp_rate.values,
    color=[GREEN,ORANGE,RED,PURPLE]
)

plt.xlabel('Chest Pain Type')
plt.ylabel('% Heart Disease')

plt.title('Chest Pain Type vs Disease')
plt.tight_layout()
plt.savefig('plots/chest_pain_vs_disease.png')
plt.close()

# =====================================================
#  Feature Importance
# =====================================================

corr_target = df.drop(columns=['target']).corrwith(df['target']).sort_values()
plt.figure(figsize=(6,6))

colors = [BLUE if v<0 else RED for v in corr_target.values]
plt.barh(corr_target.index, corr_target.values, color=colors)

plt.title("Feature Importance (Correlation)")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

# =====================================================
#  Cholesterol Distribution
# =====================================================

plt.figure(figsize = (6,4))

sns.histplot(df[df.target==0]['Cholesterol'],color=BLUE,bins=30,label='Normal')
sns.histplot(df[df.target==1]['Cholesterol'],color=RED,bins=30,label='Disease')

plt.legend()

plt.title("Cholesterol Distribution")
plt.tight_layout()
plt.savefig("plots/cholesterol_distribution.png")
plt.close()

# =====================================================
#  Vessels vs Disease
# =====================================================

vessels = df.groupby("Number of vessels fluro")["target"].mean()*100

plt.figure(figsize=(6,4))

plt.bar(
vessels.index.astype(str),
vessels.values,
color=[BLUE,ORANGE,RED,PURPLE]
)

plt.xlabel("Number of Vessels")
plt.ylabel("% Heart Disease")

plt.title("Vessels Count vs Disease")
plt.tight_layout()
plt.savefig("plots/vessels_vs_disease.png")
plt.close()