# Ex No : 01
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output:

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
# READ CSV FILE HERE
df = pd.read_csv("Loan_data.csv")
# DISPLAY THE INFORMATION ABOUT CSV AND RUN THE BASIC DATA ANALYSIS FUNCTIONS

print("\n--- First 10 rows ---")
print(df.head(10))

print("\n--- Shape (rows, cols) ---")
print(df.shape)

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Info ---")
print(df.info())

print("\n--- Summary statistics (numeric) ---")
print(df.describe())
```
<img width="592" height="840" alt="image" src="https://github.com/user-attachments/assets/e6f58552-c736-4916-b484-d694e0197226" />

```
print("\n--- Missing values per column ---")
print(df.isnull().sum())
```
<img width="380" height="360" alt="image" src="https://github.com/user-attachments/assets/7e3c646d-de70-4510-bf27-91f1c71fabfe" />


```
# DISPLAY THE SUM ON NULL VALUES IN EACH ROWS
print("\n--- Missing values per row (first 15 rows) ---")
print(df.isnull().sum(axis=1).head(15))
```
<img width="531" height="413" alt="image" src="https://github.com/user-attachments/assets/089b1ce1-8ece-4020-93c9-173a7dd0e1f3" />

 ```
# DROP NULL VALUES
df_dropna = df.dropna()
print("\n--- After dropna() shape ---")
print(df_dropna.shape)
```
<img width="423" height="73" alt="image" src="https://github.com/user-attachments/assets/21811839-24bf-446e-9f13-6b8a9a41334e" />

```
# FILL NULL VALUES WITH CONSTANT VALUE "O"
df_const = df.copy()
df_const = df_const.fillna("O")
print("\n--- After fillna('O') missing per column ---")
print(df_const.isnull().sum())
```

<img width="763" height="539" alt="image" src="https://github.com/user-attachments/assets/e6546399-4f98-4034-8d5d-ec94d7c82b4c" />

```
# FILL NULL VALUES WITH ffill or bfill METHOD:

df_ffill = df.copy().fillna(method="ffill")
df_bfill = df.copy().fillna(method="bfill")
print("\n--- After ffill() shape ---", df_ffill.shape)
print("--- After bfill() shape ---", df_bfill.shape)

```

<img width="793" height="122" alt="image" src="https://github.com/user-attachments/assets/b69f636d-c9e2-48e1-84e7-3b825d3f2a4f" />

```
# CALCULATE MEAN VALUE OF A COLUMN AND FILL IT WITH NULL VALUES:

df_mean = df.copy()

num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History"]

for c in num_cols:
    if c in df_mean.columns:
        df_mean[c] = pd.to_numeric(df_mean[c], errors="coerce")
        mean_val = df_mean[c].mean()
        df_mean[c].fillna(mean_val, inplace=True)

print("\n--- After mean-imputation, missing per column ---")
print(df_mean.isnull().sum())

```

<img width="830" height="528" alt="image" src="https://github.com/user-attachments/assets/404dac55-d4ea-47cc-9a58-2c071f5d4515" />

```
# DROP NULL VALUES (again, if you want a fully non-null dataset):

df_clean_nonull = df_mean.dropna()
print("\n--- Final non-null shape ---")
print(df_clean_nonull.shape)
```

<img width="517" height="111" alt="image" src="https://github.com/user-attachments/assets/251eebe6-8772-445c-b910-677a99c076eb" />

```
# OUTLIER DETECTION & REMOVAL:

numeric_cols = df_clean_nonull.select_dtypes(include=[np.number]).columns.tolist()
print("\n--- Numeric columns considered for outliers ---")
print(numeric_cols)
```

<img width="877" height="54" alt="image" src="https://github.com/user-attachments/assets/a51af342-e517-4385-811e-8e3a03aa1757" />

```
# USE BOXPLOT FUNCTION HERE TO DETECT OUTLIER (visual check)

for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
    if col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df_clean_nonull[col])
        plt.title(f"Boxplot before removing outliers: {col}")
        plt.show()
```

<img width="343" height="778" alt="image" src="https://github.com/user-attachments/assets/0201aae2-7b32-4122-9330-02a626f2588c" />

```

# PERFORM Z SCORE METHOD AND DETECT OUTLIER VALUES

z_threshold = 3.0
z = np.abs(stats.zscore(df_clean_nonull[numeric_cols], nan_policy="omit"))
# z is a numpy array; make a DataFrame for readability
z_df = pd.DataFrame(z, columns=numeric_cols, index=df_clean_nonull.index)

z_outlier_mask = (z_df > z_threshold).any(axis=1)
print("\n--- Total Z-score outlier rows ---", z_outlier_mask.sum())
```

<img width="350" height="35" alt="image" src="https://github.com/user-attachments/assets/cfc3ca65-6fe0-42e6-a0ba-56e0cff93a9a" />

```
# REMOVE OUTLIERS (Z-score):

df_z_clean = df_clean_nonull.loc[~z_outlier_mask].copy()
print("\n--- Shape after Z-score outlier removal ---", df_z_clean.shape)
```

<img width="624" height="49" alt="image" src="https://github.com/user-attachments/assets/02183b56-5fa4-4cf1-9d1f-7e698567484a" />

```
# USE BOXPLOT FUNCTION HERE TO CHECK OUTLIER IS REMOVED (Z-score)

for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
    if col in df_z_clean.columns:
        plt.figure()
        sns.boxplot(x=df_z_clean[col])
        plt.title(f"Boxplot after Z-score removal: {col}")
        plt.show()
```

<img width="411" height="862" alt="image" src="https://github.com/user-attachments/assets/3881ad5c-63ff-49ca-9688-9d378540989c" />


# Result
 On checking the dataset in Excel using Filter/COUNTBLANK, the following missing values were identified successfully.

 ## Summary 
In this experiment, the loan dataset was cleaned to make it suitable for analysis. Missing values (NaN) were first identified and handled using different approaches such as dropping rows, filling with constants, forward/backward fill, and mean imputation for numeric columns like ApplicantIncome and LoanAmount. These techniques helped in retaining maximum useful data instead of losing important records. Outliers were then detected through boxplots and removed using the Z-score method, which eliminated extreme values that could mislead the analysis. Through this process, I understood that data cleaning is essential for improving data quality, ensuring accuracy, and preparing a reliable dataset for further analysis or machine learning tasks.
