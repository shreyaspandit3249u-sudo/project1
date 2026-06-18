import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Student Performance Dataset
data = {
    "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Gender": ["Male", "Female", "Male", "Female", "Male",
               "Female", "Male", "Female", "Male", "Female"],
    "Age": [18, 19, 18, 20, 19, 18, 20, 19, 18, 20],
    "Attendance": [90, 95, 75, 88, 60, 92, 85, 98, 70, 89],
    "Study_Hours": [4, 5, 2, 3, 1, 5, 4, 6, 2, 3],
    "Assignment_Score": [85, 90, 70, 80, 55, 92, 84, 95, 68, 82],
    "Final_Marks": [88, 92, 72, 84, 58, 94, 86, 97, 70, 85]
}

df = pd.DataFrame(data)

# Display dataset
print("First 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Histogram of Final Marks
plt.figure(figsize=(8, 5))
plt.hist(df["Final_Marks"], bins=5)
plt.title("Distribution of Final Marks")
plt.xlabel("Final Marks")
plt.ylabel("Frequency")
plt.show()

# Attendance vs Final Marks
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Attendance", y="Final_Marks")
plt.title("Attendance vs Final Marks")
plt.show()

# Study Hours vs Final Marks
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Study_Hours", y="Final_Marks")
plt.title("Study Hours vs Final Marks")
plt.show()

# Gender-wise Performance
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Gender", y="Final_Marks")
plt.title("Gender vs Final Marks")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Outlier Detection
plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Final_Marks"])
plt.title("Outlier Detection in Final Marks")
plt.show()

# Findings
print("\nFindings:")
print("1. Students with higher attendance generally score better marks.")
print("2. Study hours have a positive impact on final marks.")
print("3. Assignment scores are strongly related to final performance.")
print("4. Correlation analysis helps identify important relationships.")
print("5. Boxplots help in detecting outliers.")