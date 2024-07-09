import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA

# Load and preprocess the dataset
file_path = r'D:\inHouseProject\raw_sales.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)
data['datesold'] = pd.to_datetime(data['datesold'])

# Encode categorical variable 'propertyType'
label_encoder = LabelEncoder()
data['propertyType'] = label_encoder.fit_transform(data['propertyType'])

# Scale numerical features
scaler = StandardScaler()
data[['postcode', 'price', 'bedrooms']] = scaler.fit_transform(data[['postcode', 'price', 'bedrooms']])

# Prepare data for clustering
data_for_clustering = data.drop(columns=['datesold'])

# Apply K-means clustering
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster_kmeans'] = kmeans.fit_predict(data_for_clustering)

# Apply K-medoids clustering
kmedoids = KMedoids(n_clusters=optimal_clusters, metric='manhattan', random_state=42, init='k-medoids++')
data['cluster_kmedoids'] = kmedoids.fit_predict(data_for_clustering)

# Streamlit app layout
st.title('Clustering Visualization')

if st.button('Update Clusters'):
    user_feedback = {
        2: {1: [1, 2, 3]}
    }

    for current_cluster, feedback in user_feedback.items():
        for target_cluster, indices in feedback.items():
            data.loc[indices, 'cluster_kmeans'] = target_cluster
            data.loc[indices, 'cluster_kmedoids'] = target_cluster

    # Re-fit K-means and K-medoids after updating clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, init=kmeans.cluster_centers_)
    data['cluster_kmeans'] = kmeans.fit_predict(data_for_clustering)
    kmedoids = KMedoids(n_clusters=optimal_clusters, metric='manhattan', random_state=42, init='k-medoids++')
    data['cluster_kmedoids'] = kmedoids.fit_predict(data_for_clustering)
    
    st.success('Clusters updated')

# Perform PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data_for_clustering)
data['pca1'] = reduced_data[:, 0]
data['pca2'] = reduced_data[:, 1]

st.subheader('PCA Visualization')
st.write(data[['pca1', 'pca2', 'cluster_kmeans', 'cluster_kmedoids']])
