# <p align = 'center'>Bank Customer Profiling and Segmentation</p>

### PROJECT OVERVIEW

In this case study, I am a consultant to a bank in New York City. The bank has extensive data on their customers for the past 6 months. The marketing team at the bank wants to launch a targeted ad marketing campaign by dividing their customers into at least 3 distinctive groups.

One of the key pain points for marketers is to `know their customers` and `identify their needs`. By understanding the customer, marketers can launch a targeted marketing campaign that is tailored for specific needs. If data about the customers is available, data science can be applied to perform market segmentation.

<img width="867" alt="Problem Statement" src="https://user-images.githubusercontent.com/39597515/214297948-3f5e8cae-730f-476f-a955-ea650baf405b.png">

#### What we will do - 

* In this project, you have been hired as a data scientist at a bank and you have been provided with extensive data on the bank's customers for the past 6 months.
* Data includes transactions frequency, amount, tenure..etc.
* The bank marketing team would like to leverage AI/ML to `launch a targeted marketing ad campaign` that is tailored to specific group of customers.
* In order for this campaign to be successful, the bank has to `divide its customers into at least 3 distinctive groups`.
* This process is known as `marketing segmentation` and it crucial for `maximizing marketing campaign conversion rate`.

### TABLE OF CONTENTS

| Sr No |         Topic                 | 
| ------| ----------------------------- |
| 1.    | Data Preparation Pipeline     |
| 2.    | Feature Engineering           |
| 3.    | Feature Selection             |
| 4.    | Kmeans Clustering and PCA     |
| 5.    | Hierarchical Clustering       |
| 6.    | Advanced Clustering           |
| 7.    | Key Learnings/Takeaways       |

### DATA DESCRIPTION

The data source is collected from Kaggle - https://www.kaggle.com/arjunbhasin2013/ccdata

1. CUSTID: Identification of Credit Card holder
2. BALANCE: Balance amount left in customer's account to make purchases
3. BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
4. PURCHASES: Amount of purchases made from account
5. ONEOFFPURCHASES: Maximum purchase amount done in one-go
6. INSTALLMENTS_PURCHASES: Amount of purchase done in installment
7. CASH_ADVANCE: Cash in advance given by the user
8. PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
9. ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
10. PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
11. CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
12. CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
13. PURCHASES_TRX: Number of purchase transactions made
14. CREDIT_LIMIT: Limit of Credit Card for user
15. PAYMENTS: Amount of Payment done by user
16. MINIMUM_PAYMENTS: Minimum amount of payments made by user
17. PRC_FULL_PAYMENT: Percent of full payment paid by user
18. TENURE: Tenure of credit card service for user

### 1. DATA PREPARATION PIPELINE

So, to start with our problem, we will clean the dataset by checking for null values, handling outliers, checking for data consistency

**A) Performing Data Analysis:**
Initial exploration and descriptive statistics were performed to understand key customer behavior:

- Mean BALANCE is $1,564
- Avg. PURCHASES ~ 1,000, with ONEOFF_PURCHASES averaging $600
- TENURE is typically 11 years
- Full balance payment rate (PRC_FULL_PAYMENT) is low (~15%)

**Outliers:**
- One customer made a one-off purchase of $40,761
- Another had a cash advance of $47,137

**B) Data Visualization:**

**Missing Values:**
- Handled missing values in `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` using **mean imputation**
- Saved a heatmap of null values: `Images/missing_values_plot.png`

**Distribution Plots:**
- Used `distplot` and KDE to visualize distributions of all features
- Saved figure as: `Images/Distplot.png`

**Insights:**
- `BALANCE_FREQUENCY` ~1 (frequent updates)
- Two distinct customer groups in `PURCHASES_FREQUENCY`
- Most customers rarely use one-off or installment purchase options

**Correlation Heatmap:**
- Created heatmap to examine feature correlations
- Found strong positive correlation between:
  - `PURCHASES`, `ONEOFF_PURCHASES`, and `INSTALLMENT_PURCHASES`
  - `PURCHASES_FREQUENCY` and `PURCHASES_INSTALLMENT_FREQUENCY`

**C) Data Cleaning:**

- Dropped irrelevant column: `CUST_ID`
- Checked for and removed missing or duplicated data
- Scaled all features using **StandardScaler**
- Normalized data for improved K-Means performance

### 2. FEATURE ENGINEERING

We performed Feature Engineering to create new features or transform existing features that will improve the performance of the model. Here are some of the new features that were created.

| New Feature                  | Description                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| `BALANCE_USAGE_RATIO`        | Proportion of balance used vs credit limit                         |
| `ONEOFF_PURCHASE_RATIO`      | Proportion of one-off purchases out of total purchases             |
| `INSTALLMENT_PURCHASE_RATIO` | Proportion of installment purchases out of total purchases         |
| `TOTAL_PURCHASES`            | Sum of one-off and installment purchases (can replace `PURCHASES`) |
| `PURCHASES_PER_TRX`          | Average amount per purchase transaction                            |
| `CASH_ADVANCE_PER_TRX`       | Average amount per cash advance transaction                        |
| `PAYMENT_RATIO`              | Ratio of total payments to credit limit                            |
| `MINIMUM_PAYMENT_RATIO`      | Ratio of minimum payment to credit limit                           |
| `FULL_PAYMENT_FLAG`          | 1 if user always makes full payments, 0 otherwise                  |
| `UTILIZATION_RATIO`          | Ratio of total spent (purchases + cash advance) to credit limit    |

<img alt="Feature_Engineering" src="https://github.com/adiag321/Bank-Customer-Profiling-and-Segmentation/blob/main/Images/Feature_Engineering.png?raw=true">

### 3. FEATURE SELECTION

Feature selection is the process of selecting a subset of the most relevant features from a large number of features to use in a machine learning model. It is an important step in the machine learning pipeline as it can greatly impact the performance of the model.

**1. Recurssive feature elimination Regressor -**

![RFE_importances](https://user-images.githubusercontent.com/39597515/214297262-5a24f94d-0d76-4929-892b-7eb0a8abf262.png)

**2. LASSO Feature Importance Regressor -**

![Lasso_Feature_Imp](https://user-images.githubusercontent.com/39597515/214297527-b3845ca5-0ac3-435e-a589-11a7c741d298.png)

### 4. APPLYING CUSTERING TECHNIQUES - K-MEANS AND PRINCIPLE COMPONENT ANALYSIS (PCA)

#### 4.1 K-Means Clustering:
`K-means` is an unsupervised learning algorithm (clustering). K-means works by grouping some data points together (clustering) in an unsupervised. The algorithm groups observations with similar attribute values together by measuring the Euclidian distance between points.

**K-Means Algorithm:**
1. Choose number of clusters "K"
2. Select random K points that are going to be the centroids for each cluster
3. Assign each data point to the nearest centroid, doing so will enable us to create "K" number of clusters
4. Calculate a new centroid for each cluster 5. Reassign each data point to the new closest centroid
6. Go to step 4 and repeat.

**Evaluation Metrics:**
| Metric                           | Meaning                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| **Silhouette Score**             | Measures how well clusters are separated. Ranges from -1 to 1. **Higher = better**        |
| **Inertia**                      | Sum of squared distances to cluster centers. **Lower = better**                           |
| **ARI (Adjusted Rand Index)**    | Measures similarity between true labels and predicted clusters. **Higher = better**       |
| **NMI (Normalized Mutual Info)** | Measures how much info is shared between true and predicted clusters. **Higher = better** |

**Implementation Highlights:**
* Feature scaling using StandardScaler for normalization
* Elbow method to identify optimal k
* Applied KMeans on the original scaled dataset
* Cluster labels appended to the original data
* Interpretation of each cluster based on customer behavior

**4.2 PCA + K-Means Clustering**
We also applied Principal Component Analysis (PCA) to reduce dimensionality and then re-ran K-Means for improved performance and interpretability.

**PCA Highlights:**

![PCA](Images/PCA.jpeg)

* PCA with 14 components retained > 90% of variance
* PCA with 3 components used for visualization and low-dimension clustering
* Cluster labels generated on PCA-transformed data

**Cluster Comparison:**
* Adjusted Rand Index (ARI): Measures similarity between clustering results
* Normalized Mutual Information (NMI): Measures shared information

| Approach | Silhouette | Inertia | ARI | NMI |
| ------------------ | ---------- | --------- | ----- | ----- |
| **Original** | 0.171 | 125665.82 | — | — |
| **PCA (14 comps)** | 0.183 | 105474.41 | 0.569 | 0.687 |
| **PCA (3 comps)** | 0.289 | 25465.82 | 0.338 | 0.456 |

1. Silhouette Score:
* Best is PCA (3 components) → 0.289
* Higher Silhouette means better separation and cohesion.

2. Inertia:
* Lowest (i.e., best) again for PCA (3 components) → 25,465.82
* Indicates tighter clusters.

3. ARI & NMI:
* PCA (3) captures compact clusters, but loses alignment with the original label structure. (ARI = 0.338, NMI = 0.456)
* PCA (14) preserves more meaningful information for matching true labels. (ARI = 0.569, NMI = 0.687)

**Recommendation:**

Based on evaluation metrics, clustering using PCA with 3 components is optimal. It significantly improves Silhouette Score and Inertia, while still maintaining a `decent` match with original clusters (via ARI/NMI). This suggests PCA with 3 components enhances clustering performance and interpretability.

## 5. HIERARCHICAL CLUSTERING


## 6. ADVANCED CLUSTERING

## RESOURCES

https://www.analyticsvidhya.com/blog/2021/03/customer-profiling-and-segmentation-an-analytical-approach-to-business-strategy-in-retail-banking/


### 7. KEY LEARNINGS/TAKEAWAYS

* **Dimensionality Reduction via PCA** significantly improved clustering performance:
  * Reduced noise and redundancy from high-dimensional data.
  * Made clusters more separable, especially with 3 principal components.

* **Silhouette Score** emerged as the most important metric in this unsupervised context:
  * Even though PCA with 3 components captured less variance, it achieved the **highest silhouette score (0.332)** and **lowest inertia**, indicating **well-separated and tight clusters**.
  * **Goal was to create meaningful clusters**, not necessarily preserve all data variance.

* Learned about and applied **ARI (Adjusted Rand Index)** and **NMI (Normalized Mutual Information)**:
  * Helped measure how well PCA-based clusters aligned with the original KMeans clusters.
  * **High ARI/NMI for PCA with 14 components** showed better alignment with original labels, but not better clustering quality.

* **PCA with fewer components (3)** can still yield better clustering if the objective is natural separation rather than label alignment.

* The **final recommendation** to use PCA with 3 components was based on a tradeoff: better clustering quality (Silhouette) vs. perfect label preservation (ARI/NMI).



