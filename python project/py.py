import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Merged_Annually_Quarterly.csv")

# Display first rows
print("Dataset Preview:")
print(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Find missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Total missing values
print("\nTotal Missing Values:", df.isnull().sum().sum())

# ---------------------------
# Handling Missing Data
# ---------------------------

# 1. Fill numeric columns with mean
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 2. Fill categorical columns with mode (FIXED)
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Verify no missing values left
print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nData cleaned and saved successfully!")

# -------------------------------
# Data Visualization
# -------------------------------

# 1. Histogram for all numeric columns
df[numeric_cols].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 3. Pairplot (for smaller datasets)
sns.pairplot(df[numeric_cols])
plt.show()

# 4. Boxplot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Features")
plt.show()

# 5. Countplot for categorical columns (first one)
if len(categorical_cols) > 0:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df[categorical_cols[0]])
    plt.xticks(rotation=45)
    plt.title(f"Count Plot of {categorical_cols[0]}")
    plt.show()

# -------------------------------
# Feature Scaling (Sklearn)
# -------------------------------
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nScaled Data Sample:")
print(df_scaled.head())

#Trend over Time (VERY IMPORTANT)
df_grouped = df.groupby('year')['current_price'].mean()

plt.figure(figsize=(10,5))
df_grouped.plot(marker='o')
plt.title("Trend of Current Price Over Years")
plt.xlabel("Year")
plt.ylabel("Current Price")
plt.show()

#Industry-wise Comparison
industry_data = df.groupby('industry')['current_price'].mean().sort_values(ascending=False)

industry_data.head(10).plot(kind='bar', figsize=(10,5))
plt.title("Top Industries by Current Price")
plt.show()


#Distribution of Data
plt.hist(df['current_price'], bins=50)
plt.title("Distribution of Current Price")
plt.show()

#Sector-wise Contribution
sector_data = df.groupby('institutional_sector')['current_price'].sum()

sector_data.plot(kind='pie', autopct='%1.1f%%')
plt.title("Sector Contribution")
plt.ylabel("")
plt.show()

# Industry performance (create this FIRST)
industry_perf = df.groupby('industry')['current_price'].sum().sort_values(ascending=False)

# Plot
plt.figure(figsize=(12,6))
industry_perf.head(10).plot(kind='barh')
plt.title("Top 10 Performing Industries")
plt.xlabel("Total Current Price")
plt.ylabel("Industry")
plt.gca().invert_yaxis()
plt.show()

