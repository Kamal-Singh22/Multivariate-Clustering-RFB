import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Function to load and preprocess data
@st.cache(allow_output_mutation=True)
def load_and_preprocess_data(file_path, sample_size=10):
    data = pd.read_csv(file_path)
    
    # Take the first 20 data points
    data = data.head(sample_size)
    
    # Preprocess the data
    data['datesold'] = pd.to_datetime(data['datesold'])
    label_encoder = LabelEncoder()
    data['propertyType'] = label_encoder.fit_transform(data['propertyType'])
    scaler = StandardScaler()
    data[['postcode', 'price', 'bedrooms']] = scaler.fit_transform(data[['postcode', 'price', 'bedrooms']])
    
    return data

# Function to find optimal clusters using the elbow method
def find_optimal_clusters(data, max_k):
    iters = range(1, max_k + 1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, max_iter=20, random_state=42, n_init=5)
        kmeans.fit(data[['postcode', 'price', 'bedrooms']])
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal Number of Clusters')
    st.pyplot(plt)

# Function to apply K-means clustering
def apply_kmeans(data, optimal_clusters):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data[['postcode', 'price', 'bedrooms']])
    
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    visualize_data = pd.DataFrame(X_pca, columns=['pca1', 'pca2'])
    visualize_data['cluster_kmeans'] = cluster_labels
    visualize_data['original_index'] = data.index  # Keep track of original indices
    
    return visualize_data, kmeans, cluster_labels

# Main function to run the Streamlit app
def main():
    st.title('Real Estate Sales Clustering Visualization')

    # Load and preprocess data
    file_path = 'raw_sales.csv'  # Replace with your actual file path
    data_load_state = st.text('Loading data...')
    data = load_and_preprocess_data(file_path)
    data_load_state.text('Data loaded successfully!')

    # Display some sample data
    st.subheader('Sample of Preprocessed Data')
    st.write(data.head(20))

    # Determine the optimal number of clusters
    st.subheader('Find Optimal Clusters')
    max_clusters = st.slider('Select maximum number of clusters:', 2, 10, 4)
    find_optimal_clusters(data[['postcode', 'price', 'bedrooms']], max_clusters)

    # Choose the optimal number of clusters
    optimal_clusters = st.number_input('Enter the optimal number of clusters:', min_value=2, max_value=10, value=4)

    # Apply K-means clustering and visualize clusters using PCA
    st.subheader('Cluster Visualization')
    visualize_data, kmeans_model, cluster_labels = apply_kmeans(data, optimal_clusters)
    scatter_chart = st.scatter_chart(data=visualize_data, x='pca1', y='pca2', color='cluster_kmeans')

    # Section for user feedback to update clusters
    st.write("Use the slider and input box below to manually update cluster assignments.")

    # Slider for selecting data point index
    data_point_index = st.slider("Select Data Point Index:", 0, len(visualize_data)-1, 0)

    # Input box for specifying the new cluster
    new_cluster = st.number_input("Enter New Cluster Number:", min_value=0, max_value=optimal_clusters-1, value=0)

    # Button to update cluster assignment
    if st.button('Assign Cluster'):
        # Update cluster label in the dataframe
        visualize_data.loc[visualize_data['original_index'] == data_point_index, 'cluster_kmeans'] = new_cluster
        st.success(f"Data point at index {data_point_index} assigned to cluster {new_cluster}.")

        # Update scatter plot with new data
        scatter_chart.data = visualize_data

        # Update scatter chart display
        st.subheader('Updated Cluster Visualization')
        st.scatter_chart(data=visualize_data, x='pca1', y='pca2', color='cluster_kmeans')

        # Display data points and their cluster assignments
        st.subheader('Data Points and Cluster Assignments')
        st.write(visualize_data[['original_index', 'cluster_kmeans']].head(20))  # Display first 20 rows
        

if __name__ == '__main__':
    main()
