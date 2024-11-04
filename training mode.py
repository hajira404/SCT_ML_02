import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
data = pd.read_csv("customer_data.csv")

# Selecting the relevant features
features = data[['age', 'gender', 'income', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category', 'purchase_amount', 'satisfaction_score']].copy()

# Step 1: Encode categorical variables
categorical_features = ['gender', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category']
for col in categorical_features:
    label_encoder = LabelEncoder()
    features.loc[:, col] = label_encoder.fit_transform(features[col])

# Step 2: Scaling numerical data
scaler = StandardScaler()
numerical_features = ['age', 'income', 'purchase_amount', 'satisfaction_score']
features.loc[:, numerical_features] = scaler.fit_transform(features[numerical_features])

# Step 3: Using the Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to find optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 4: Choosing the optimal number of clusters based on the Elbow graph
# Set the number of clusters to your chosen optimal value
optimal_clusters = 4  # Adjust this value if needed

# Step 5: Applying K-Means with the optimal number of clusters on 'income' and 'purchase_amount'
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)

# Perform clustering using only 'income' and 'purchase_amount' columns
data['Cluster'] = kmeans.fit_predict(features[['income', 'purchase_amount']])

# Step 6: Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='income', y='purchase_amount', hue='Cluster', data=data, palette='viridis', s=60, edgecolor='k')
plt.scatter(
    kmeans.cluster_centers_[:, 0], 
    kmeans.cluster_centers_[:, 1], 
    s=300, 
    c='red', 
    marker='X', 
    label='Cluster Centers'
)
plt.title('Customer Segments Based on Income and Purchase Amount')
plt.xlabel('Annual Income')
plt.ylabel('Purchase Amount')
plt.legend(title='Cluster')
plt.show()

# Grouping by cluster and calculating the mean for numeric features only
cluster_summary = data.groupby('Cluster').mean(numeric_only=True)  # Use numeric_only=True for recent Pandas versions
print(cluster_summary)