# 📊 E-Commerce Customer Churn Prediction & Segmentation | Machine Learning | Python

**Business Question:** Which customers are likely to churn — and among those who do, what behavioral sub-groups exist to enable targeted retention?

**Domain:** E-Commerce / Customer Retention / Machine Learning

**Tools Used:** Python (Scikit-learn, Pandas, Matplotlib, Seaborn)

> 📌 This project tackles customer churn from **two complementary angles** using the same `churn_prediction.xlsx` dataset:
> - **Supervised Learning** — Predict *who* will churn using classification models
> - **Unsupervised Learning** — Discover *why* they churn by clustering churned customers into behavioral sub-groups
>
> Together, these two approaches form a complete churn intelligence pipeline: identify at-risk customers, then personalize retention strategies based on their cluster profile.

**Author:** Phan Trung Hiếu
**Date:** 2024

---

## 📑 Table of Contents

1. [📌 Background & Overview](#-background--overview)
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)
3. [⚒️ Main Process](#️-main-process)
   - [Part A — Supervised Learning: Churn Prediction](#part-a--supervised-learning-churn-prediction)
   - [Part B — Unsupervised Learning: Churned Customer Segmentation](#part-b--unsupervised-learning-churned-customer-segmentation)
4. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

---

## 📌 Background & Overview

### 📖 What is this project about? What Business Question will it solve?

This project uses **Python & Scikit-learn** to analyze customer transaction and behavioral data from an e-commerce platform to:

✔️ **[Supervised]** Build a classification model to predict whether a customer will churn — enabling proactive intervention before the customer leaves.

✔️ **[Supervised]** Compare multiple classification algorithms (Logistic Regression, KNN, Random Forest) and select the best-performing model.

✔️ **[Unsupervised]** Apply PCA and K-Means clustering exclusively on churned customers to discover distinct behavioral sub-groups.

✔️ **[Unsupervised]** Identify the most important features that differentiate churn clusters — providing actionable insight into *what drives each group's churn*.

✔️ Connect both analyses to form a unified retention strategy: predict churn early, then route each at-risk customer to the most appropriate intervention based on their cluster.

### 👤 Who is this project for?

✔️ **CRM & Retention teams** — to identify at-risk customers and tailor win-back strategies per behavioral cluster.

✔️ **E-commerce product managers** — to understand which product/UX factors correlate most with churn.

✔️ **Data scientists & ML practitioners** — to see how supervised and unsupervised approaches complement each other on the same business problem.

---

## 📂 Dataset Description & Data Structure

### 📌 Data Source

- **Source:** `churn_prediction.xlsx` — E-commerce customer behavioral dataset
- **Size:** 5,630 rows × 20 columns
- **Format:** `.xlsx`
- **Target Variable:** `Churn` (binary: 0 = retained, 1 = churned)
- **Missing Data:** ~32.97% of rows contain at least one missing value

### 📊 Data Structure & Relationships

#### 1️⃣ Tables Used:

A single flat table. For Supervised Learning, all 5,630 rows are used. For Unsupervised Learning, only the **948 churned customers** (`Churn == 1`) are used.

#### 2️⃣ Table Schema & Data Snapshot

<img width="1728" height="195" alt="image" src="https://github.com/user-attachments/assets/b28bcad5-9ac8-4ee9-895a-4d12e5fa1a57" />

| Column Name | Data Type | Description |
|---|---|---|
| `CustomerID` | INT | Unique customer identifier |
| `Churn` | INT | Target: 1 = churned, 0 = retained |
| `Tenure` | FLOAT | Months with the platform (7.0% missing) |
| `PreferredLoginDevice` | STRING | Mobile Phone / Computer |
| `CityTier` | INT | City tier classification (1, 2, 3) |
| `WarehouseToHome` | FLOAT | Distance from warehouse to customer (4.6% missing) |
| `PreferredPaymentMode` | STRING | E-wallet, Debit Card, COD, UPI, etc. |
| `Gender` | STRING | Male / Female |
| `HourSpendOnApp` | FLOAT | Avg hours on app per month (4.5% missing) |
| `NumberOfDeviceRegistered` | INT | Devices linked to account |
| `PreferedOrderCat` | STRING | Preferred product category |
| `SatisfactionScore` | INT | Customer satisfaction (1–5) |
| `MaritalStatus` | STRING | Married / Single / Divorced |
| `NumberOfAddress` | INT | Number of saved addresses |
| `Complain` | INT | 1 if filed a complaint in last month |
| `OrderAmountHikeFromlastYear` | FLOAT | % increase in orders vs. last year (4.7% missing) |
| `CouponUsed` | FLOAT | Coupons used last month (11.17% outliers) |
| `OrderCount` | FLOAT | Orders placed last month (12.49% outliers) |
| `DaySinceLastOrder` | FLOAT | Days since most recent order (5.5% missing) |
| `CashbackAmount` | FLOAT | Avg cashback received (7.78% outliers) |

---

## ⚒️ Main Process

### Shared Preprocessing (Both Approaches)

Both notebooks share the same initial cleaning steps:

**1. Missing Value Treatment** — Numerical columns filled with **median**; categorical columns filled with **mode** (safe, non-leaking strategy).

```python
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)
```

**2. Outlier Check (IQR method)** — Notable outlier rates:

| Feature | Outlier % |
|---|---|
| `CouponUsed` | 11.17% |
| `OrderCount` | 12.49% |
| `CashbackAmount` | 7.78% |
| `NumberOfDeviceRegistered` | 7.05% |

**3. Duplicate Check** — No duplicate rows found (`df.duplicated().any()` → `False`).

**4. One-Hot Encoding** — 5 categorical columns encoded using `pd.get_dummies(drop_first=True)`:
`PreferredLoginDevice`, `PreferredPaymentMode`, `Gender`, `PreferedOrderCat`, `MaritalStatus`

<img width="973" height="740" alt="image" src="https://github.com/user-attachments/assets/48248ca1-e5fe-41ca-9755-eb29f0198a39" />

---

### Part A — Supervised Learning: Churn Prediction

#### Step 1: Train / Test Split

```python
x = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

| Set | Records |
|---|---|
| Train | 4,504 (80%) |
| Test | 1,126 (20%) |

#### Step 2: Standardization

`StandardScaler` was fitted **only on training data** and applied to both train and test — to prevent data leakage.

```python
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)  # transform only — no fit
```

#### Step 3: Model Comparison

Three classification models were trained and evaluated on test accuracy:

```python
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest'      : RandomForestClassifier(),
    'KNN'               : KNeighborsClassifier()
}
```

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 88.37% |
| KNN | 89.25% |
| **Random Forest** | **96.36% ✅** |

> 📌 **Random Forest** was selected as the best model due to its significantly higher accuracy, robustness to outliers, and ability to handle non-linear relationships without feature scaling assumptions.

#### Step 4: Model Evaluation — Random Forest

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)
```

| Metric | Train | Test |
|---|---|---|
| **Accuracy** | 100% | **96%** |
| Precision (Class 0) | 1.00 | — |
| Recall (Class 0) | 1.00 | — |
| Precision (Class 1) | 1.00 | — |
| Recall (Class 1) | 1.00 | — |

> 📌 Train accuracy of 100% indicates the model has memorized the training data (overfitting). The gap between train (100%) and test (96%) suggests there is room to improve generalization — e.g., via `max_depth` regularization, cross-validation, or hyperparameter tuning with `GridSearchCV`.

<img width="553" height="351" alt="image" src="https://github.com/user-attachments/assets/cb332645-ea35-4b98-978c-df4586f4f5be" />

---

### Part B — Unsupervised Learning: Churned Customer Segmentation

> **Scope:** Only the **948 churned customers** (`Churn == 1`) are analyzed here. The goal is not to predict churn, but to understand the behavioral sub-types *within* churned customers.

#### Step 1: Filter & Encode

```python
df = df[df['Churn'] == 1].reset_index(drop=True)
# One-hot encode same 5 categorical columns, drop CustomerID
df_encoded = pd.get_dummies(df, columns=list_encode_cols, drop_first=True)
df_encoded = df_encoded.drop(columns=['CustomerID'])
```

#### Step 2: Dimensionality Reduction — PCA

PCA reduced the feature space to **3 principal components** before clustering:

```python
pca = PCA(n_components=3)
PCA_ds = pd.DataFrame(pca.transform(df_encoded), columns=["col1","col2","col3"])
```

| Component | Variance Explained |
|---|---|
| PC1 | **91.20%** |
| PC2 | 4.24% |
| PC3 | 1.71% |
| **Total** | **97.15%** |

> 📌 PC1 captures over 91% of the variance, meaning most of the information in the dataset can be represented in a single dimension. This suggests strong dominant patterns in churned customer behavior.

#### Step 3: Choosing K — Elbow Method

```python
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(PCA_ds)
    ss.append(kmeans.inertia_)
```

<img width="841" height="467" alt="image" src="https://github.com/user-attachments/assets/6f544157-0785-443f-84cf-b1ef4557f717" />

The elbow curve indicated **k = 3** as the optimal number of clusters.

#### Step 4: K-Means Clustering

```python
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
predicted_labels = kmeans.fit_predict(PCA_ds)
```

#### Step 5: Cluster Evaluation

**Silhouette Score: 0.558** — a moderate-to-good score, indicating reasonably well-separated clusters that are meaningful for downstream analysis.

To further validate cluster separability, a **Random Forest classifier** was trained on the cluster labels:

```python
clf_rand = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=100)
# Balanced Accuracy Score on test set: 0.997
```

> 📌 A balanced accuracy of **99.7%** on predicting cluster labels confirms the 3 clusters are highly distinct and internally cohesive — validating the quality of the K-Means segmentation.

#### Step 6: Cluster Analysis

**Cluster size distribution:**

| Cluster | Count |
|---|---|
| Cluster 0 | 34 |
| Cluster 1 | 673 |
| Cluster 2 | 241 |

> 📌 Cluster 1 is the dominant group (~71% of churned customers). Cluster 0 is a small, potentially high-signal outlier group worth investigating closely.

<img width="567" height="454" alt="image" src="https://github.com/user-attachments/assets/a7297404-53b6-466a-8c05-f1e37e6ce424" />


<img width="569" height="430" alt="image" src="https://github.com/user-attachments/assets/dafaa79b-b506-4a64-93f0-b55c0836f37e" />


**Top Features by Gini Importance (Random Forest on clusters):**

<img width="856" height="663" alt="image" src="https://github.com/user-attachments/assets/5770ce8e-f0ed-478c-a4d0-d99b091ff925" />


> 📌 CashbackAmount emerged as a key differentiator between clusters, suggesting that cashback incentive levels play a significant role in the type of churn behavior customers exhibit.

---

## 🔎 Final Conclusion & Recommendations

Based on the combined supervised and unsupervised analysis, we recommend the **CRM & Marketing team** to consider the following:

✔️ **Deploy the Churn Prediction Model in Production** — The Random Forest model achieves **96% test accuracy**. Integrate it into a weekly scoring pipeline to flag at-risk customers before they churn, allowing proactive outreach.

✔️ **Address Model Overfitting** — The perfect train accuracy (100%) vs 96% test accuracy signals overfitting. Tune `max_depth`, apply cross-validation, or test ensemble methods like XGBoost to improve generalization before production deployment.

✔️ **Personalize Retention by Cluster** — Instead of one-size-fits-all win-back campaigns, use the 3 churned customer clusters to design differentiated strategies:
- **Cluster 0 (34 customers)** — Small, potentially high-value or highly distinct segment. Investigate manually — may warrant a premium personalized outreach.
- **Cluster 1 (673 customers)** — The majority churn group. Scalable retention campaigns (e.g., cashback boost, satisfaction surveys) would have the broadest impact here.
- **Cluster 2 (241 customers)** — Mid-size segment with distinct behavioral patterns. Targeted interventions based on their top differentiating features.

✔️ **Leverage CashbackAmount as a Retention Lever** — CashbackAmount is a top feature distinguishing clusters. Consider running A/B tests on cashback rate adjustments for at-risk customer segments to measure uplift in retention.

✔️ **Investigate Complaint-Churn Relationship** — The `Complain` feature and `SatisfactionScore` are strong churn signals. A fast complaint resolution SLA and proactive follow-up on low satisfaction scores can meaningfully reduce churn.

✔️ **Handle Missing Data More Rigorously** — With 32.97% of rows having at least one missing value, median/mode imputation may introduce bias. Consider model-based imputation (e.g., KNN Imputer) or flagging missingness as a feature in future iterations.


- 👤 Điền tên vào `[Your Name]`
- 📸 Chèn các screenshot từ notebook vào chỗ `👉🏻 Insert ... here`
- (Tùy chọn) Điền thêm số liệu cụ thể từ Confusion Matrix và Feature Importance chart sau khi chạy lại notebook
