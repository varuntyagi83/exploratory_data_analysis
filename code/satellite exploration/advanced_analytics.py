import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import requests
from google.colab import files
import os
from io import BytesIO
from urllib.parse import urlparse
import seaborn as sns

# URL of the Excel file
file_url = "https://www.ucsusa.org/sites/default/files/2024-01/UCS-Satellite-Database%205-1-2023.xlsx"

# Extract the filename from the URL
file_name = os.path.basename(urlparse(file_url).path)

# Check if the file already exists in the Colab content folder
if not os.path.exists(file_name):
    # Send a request to the URL
    response = requests.get(file_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Read the content of the response into a BytesIO object
        file_content = BytesIO(response.content)
        
        # Load the BytesIO content into a Pandas DataFrame
        satellite_data = pd.read_excel(file_content)
        
        # Save the file in the Colab content folder with the original name
        satellite_data.to_excel(file_name, index=False)
        
        # Download the file
        #files.download(file_name)
        print(f"File downloaded successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
else:
    print(f"The file '{file_name}' already exists in the Colab content folder.")


satellite_data = satellite_data.loc[:, ~satellite_data.columns.str.contains('^Unnamed')]

# Fill NaN values with the mean of each column
satellite_data = satellite_data.fillna(satellite_data.mean(numeric_only=True))

# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(satellite_data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Clustering
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime

# Cluster Analysis
numerical_columns = satellite_data.select_dtypes(include=[np.number])

# Exclude 'Power (watts)' from the features used during scaling
features_for_scaling = ['Launch Mass (kg.)', 'Period (minutes)', 'NORAD Number', 'Perigee (km)', 'Apogee (km)', 'Eccentricity', 'Inclination (degrees)', 'Longitude of GEO (degrees)', 'Cluster']
numerical_columns = numerical_columns[features_for_scaling]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_columns)

# Explicitly set n_init to avoid warning
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
satellite_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize Clusters in 2D Space using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=satellite_data['Cluster'], palette='viridis')
plt.title('Clusters Visualized in 2D Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Function to generate synthetic satellite data with consistent feature names
def generate_synthetic_data(features, cluster_centers):
    np.random.seed(42)
    num_samples = 500

    # Generate cluster labels uniformly
    cluster_labels = np.random.choice([0, 1, 2], num_samples)

    # Generate synthetic data points around the corresponding cluster centers
    synthetic_data = {feature: [] for feature in features}
    for feature in features:
        if feature == 'Cluster':
            synthetic_data[feature] = cluster_labels
        else:
            synthetic_data[feature] = np.random.normal(loc=cluster_centers[cluster_labels, features.index(feature)], scale=10, size=num_samples)

    # Create DataFrame with the specified feature order
    synthetic_satellite = pd.DataFrame(synthetic_data, columns=features)

    return synthetic_satellite


# Classify Synthetic Data into Identified Clusters
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Split the original dataset into features (X) and target variable (y)
X = scaled_data
y = satellite_data['Cluster']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict clusters for the synthetic satellite data
synthetic_cluster_centers = kmeans.cluster_centers_  # Use the actual cluster centers from the dataset
synthetic_satellite = generate_synthetic_data(features_for_scaling, synthetic_cluster_centers)
scaled_synthetic_data = scaler.transform(synthetic_satellite)
synthetic_satellite['Predicted Cluster'] = classifier.predict(scaled_synthetic_data)

# Evaluate the accuracy on the test set
test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
print(f"Test Accuracy: {test_accuracy}")

# Visualize Clusters in 2D Space using PCA with Classified Synthetic Data
pca_result_synthetic = pca.transform(scaled_synthetic_data)

# Scatter plot with cluster visualization
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=satellite_data['Cluster'], palette='viridis')

# Plot synthetic satellite data within identified clusters
for cluster_label in range(3):
    cluster_mask = synthetic_satellite['Predicted Cluster'] == cluster_label
    plt.scatter(x=pca_result_synthetic[cluster_mask, 0], y=pca_result_synthetic[cluster_mask, 1],
                c=f'C{cluster_label}', s=100, marker='o', label=f'Synthetic Cluster {cluster_label}')

plt.title('Clusters Visualized in 2D Space with Classified Synthetic Satellite Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Network Analysis
import networkx as nx

# Load world map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a mapping between countries and continents
country_continent_mapping = dict(zip(world['name'], world['continent']))

# Create a new 'Continent' column based on the mapping
satellite_data['Continent'] = satellite_data['Country of Operator/Owner'].map(country_continent_mapping)

# Network Analysis
G = nx.Graph()

# Add nodes (countries) with their continents as attributes
for country, continent in zip(satellite_data['Country of Operator/Owner'], satellite_data['Continent']):
    G.add_node(country, continent=continent)

# Add edges (connections between countries)
for _, row in satellite_data.iterrows():
    G.add_edge(row['Country of Operator/Owner'], row['Launch Site'])

# Plot the network
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42)  # Seed for reproducibility
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10)
plt.title('Network Analysis of Satellite Launches by Continent')
plt.show()


# Sankey Diagram
import plotly.graph_objects as go

# Convert NaN values in the 'Continent' column to 'Others'
satellite_data['Continent'] = satellite_data['Continent'].fillna('Others')

# Create a Sankey diagram based on the flow of data between Purposes and Continents
unique_purposes = satellite_data['Purpose'].unique()
unique_continents = satellite_data['Continent'].unique()

# Create a mapping from purposes and continents to unique integer indices
node_dict = {node: idx for idx, node in enumerate(unique_purposes)}
node_dict.update({node: idx + len(unique_purposes) for idx, node in enumerate(unique_continents)})

# Map the Purpose and Continent columns to the integer indices
satellite_data['Purpose_Index'] = satellite_data['Purpose'].map(node_dict)
satellite_data['Continent_Index'] = satellite_data['Continent'].map(node_dict)

# Create the source, target, and value arrays for the Sankey diagram
sources = satellite_data['Purpose_Index'].tolist()
targets = satellite_data['Continent_Index'].tolist()
values = [1] * len(sources)  # Assign a value of 1 for each link

# Create a color scale based on the unique indices of sources and targets
color_scale = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
unique_indices = set(sources + targets)
color_mapping = {idx: color_scale[i % len(color_scale)] for i, idx in enumerate(unique_indices)}

# Create the Sankey diagram with improved layout
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=unique_purposes.tolist() + unique_continents.tolist()
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=[color_mapping[idx] for idx in sources]
    )
)])
fig.update_layout(
    title_text='Sankey Diagram of Satellite Purposes and Continents',
    font_size=12,
    height=500,  # Adjust the height of the diagram
)
fig.show()
