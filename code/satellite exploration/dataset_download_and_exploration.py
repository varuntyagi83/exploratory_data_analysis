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

satellite_data.head()

satellite_data = satellite_data.loc[:, ~satellite_data.columns.str.contains('^Unnamed')]

# Fill NaN values with the mean of each column
satellite_data = satellite_data.fillna(satellite_data.mean(numeric_only=True))

# Set a larger plot size for better visualization
plt.figure(figsize=(14, 10))

# 1. Distribution of Satellite Launch Mass
plt.subplot(3, 2, 1)
sns.histplot(satellite_data['Launch Mass (kg.)'], bins=20, kde=True)
plt.title('Distribution of Satellite Launch Mass')
plt.xlabel('Launch Mass (kg.)')
plt.ylabel('Frequency')

# 2. Orbit Analysis
plt.subplot(3, 2, 2)
sns.countplot(y='Class of Orbit', data=satellite_data, order=satellite_data['Class of Orbit'].value_counts().index, hue = 'Class of Orbit')
plt.title('Distribution of Satellite Orbits (Class)')
plt.xlabel('Count')
plt.ylabel('Class of Orbit')

# 3. Temporal Analysis
plt.subplot(3, 2, 3)
satellite_data['Date of Launch'] = pd.to_datetime(satellite_data['Date of Launch'], errors='coerce')  # handle invalid dates
sns.histplot(satellite_data['Date of Launch'].dt.year.dropna(), bins=len(satellite_data['Date of Launch'].dt.year.dropna().unique()))
plt.title('Distribution of Satellite Launches Over Time')
plt.xlabel('Year')
plt.ylabel('Frequency')

# 4. Contractor Analysis
plt.subplot(3, 2, 4)
top_contractors = satellite_data['Contractor'].value_counts().head(10)  # Display top 10 contractors
sns.barplot(y=top_contractors.index, x=top_contractors.values, palette='viridis', hue = top_contractors.index)
plt.title('Top 10 Contractors for Satellite Missions')
plt.xlabel('Count')
plt.ylabel('Contractor')

# 5. Country-Based Analysis
plt.subplot(3, 2, 5)
top_countries = satellite_data['Country of Operator/Owner'].value_counts().head(10)  # Display top 10 countries
sns.barplot(y=top_countries.index, x=top_countries.values, palette='viridis', hue=top_countries.index )
plt.title('Top 10 Countries for Satellite Missions')
plt.xlabel('Count')
plt.ylabel('Country')

# 6. Lifetime Analysis
plt.subplot(3, 2, 6)
sns.histplot(satellite_data['Expected Lifetime (yrs.)'].dropna(), bins=20, kde=True)
plt.title('Distribution of Expected Lifetime for Satellites')
plt.xlabel('Expected Lifetime (years)')
plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()
