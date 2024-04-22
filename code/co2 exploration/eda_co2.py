import pandas as pd
import zipfile
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Download the ZIP file
url = "https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.PC?start=2000&end=2023&downloadformat=csv"
response = requests.get(url)

# Save the ZIP file to a temporary file
with open("co2_emissions.zip", "wb") as zip_file:
    zip_file.write(response.content)

# Unzip the file
with zipfile.ZipFile("co2_emissions.zip", "r") as zip_ref:
    zip_ref.extractall()

# Delete the ZIP file
import os
os.remove("co2_emissions.zip")

# Load the CSV data into a DataFrame
df = pd.read_csv("API_EN.ATM.CO2E.PC_DS2_en_csv_v2_191.csv", skiprows=4)

df.head(10)

# Melt the DataFrame to a format suitable for analysis
melted_df = df.melt(id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"], var_name="Year", value_name="CO2 Emissions (metric tons per capita)")

# Handle missing values (replace with appropriate methods based on your analysis)
melted_df.dropna(subset=["CO2 Emissions (metric tons per capita)"], inplace=True)

# Explore basic information (optional)
print("Number of rows:", melted_df.shape[0])
print("Number of columns:", melted_df.shape[1])
print("List of columns:", melted_df.columns.tolist())
print("Data types of each column:", melted_df.dtypes)

# Print the unique values of years
print(melted_df['Year'].unique())

print(melted_df['Indicator Name'].unique())

print(melted_df['CO2 Emissions (metric tons per capita)'].unique())

melted_df.head(10)

# Select the column containing CO2 emissions (assuming it's named "CO2 Emissions (metric tons per capita)")
co2_emissions_col = "CO2 Emissions (metric tons per capita)"

# Filter data for the desired country
country_name = "India"  # Replace with the desired country
country_data = melted_df[melted_df["Country Name"] == country_name]

# Plot CO2 emissions over time for the specific country
country_data.plot(x="Year", y=co2_emissions_col, marker="o")
plt.title(f"CO2 Emissions in {country_name} over Time")
plt.xlabel("Year")
plt.ylabel(co2_emissions_col)  # Add label for the y-axis
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Explore trends over time and identify major contributors
plt.figure(figsize=(20, 8))
sns.lineplot(x='Year', y=co2_emissions_col, hue='Country Name', data=melted_df)
plt.title('CO2 Emissions Trends Over Time')
plt.legend().set_visible(False)
plt.show()

# Calculate total emissions for each country
total_emissions = melted_df.groupby('Country Name')[co2_emissions_col].sum()

# Identify major contributors (e.g., top 10)
major_contributors = total_emissions.nlargest(10).index.tolist()

# Filter the data for major contributors
filtered_df = melted_df[melted_df['Country Name'].isin(major_contributors)]

# Plot the data with legends for major contributors only
plt.figure(figsize=(20, 8))
sns.lineplot(x='Year', y=co2_emissions_col, hue='Country Name', data=filtered_df)
plt.title('CO2 Emissions Trends Over Time')
plt.legend()
plt.show()

# Calculate descriptive statistics by time period (As 'Year' is the time period column)
print("Descriptive statistics by time period:")
print(melted_df.groupby('Year')['CO2 Emissions (metric tons per capita)'].describe())

# Calculate descriptive statistics by country
print("\nDescriptive statistics by Country:")
print(melted_df.groupby('Country Name')['CO2 Emissions (metric tons per capita)'].describe())

# Interactive Maps
!wget 'https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip'

# Display on the map
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from IPython.display import display

# Sum the total emissions per region/country
region_emissions = melted_df.groupby('Country Name')['CO2 Emissions (metric tons per capita)'].sum().reset_index().round(2)
region_emissions.columns = ['Country Name', 'sum']

# Load the '110m' cultural vectors dataset from a local file path
world = gpd.read_file('/content/ne_110m_admin_0_countries.zip')

# Merge the world map with the sum co2 based on ISO country codes
world = world.merge(region_emissions, left_on='ADMIN', right_on='Country Name', how='left')

# Filter out rows with count 0
world = world[world['sum'] > 0]

# Create a Folium Map centered at the mean of latitude and longitude
m = folium.Map(location=[world.geometry.centroid.y.mean(), world.geometry.centroid.x.mean()], zoom_start=2)

# Add GeoJSON data to the map
folium.GeoJson(world, name='geojson', tooltip=folium.features.GeoJsonTooltip(fields=['ADMIN', 'sum'], aliases=['Country', 'sum'])).add_to(m)

# Add sum values as text annotations using MarkerCluster
marker_cluster = MarkerCluster().add_to(m)
for idx, row in world.iterrows():
    folium.Marker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        popup=f"{row['ADMIN']}: {row['sum']:.0f}",
    ).add_to(marker_cluster)

# Display the map in the Colab notebook
display(m)

# Display on the map between specific years
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from IPython.display import display

# Define the starting and ending year of the desired time frame
start_year = 2000
end_year = 2020

# Convert the 'Year' column to integers before filtering
melted_df['Year'] = pd.to_numeric(melted_df['Year'], errors='coerce')

# Filter the DataFrame for the specified time frame
filtered_df = melted_df[(melted_df['Year'] >= start_year) & (melted_df['Year'] <= end_year)]

# Calculate the sum of CO2 emissions per region/country
region_emissions = (filtered_df.groupby('Country Name')['CO2 Emissions (metric tons per capita)']
                    .sum()
                    .reset_index()
                    .round(2)
                    )

region_emissions.columns = ['Country Name', 'sum']

# Load the '110m' cultural vectors dataset from a local file path
world = gpd.read_file('/content/ne_110m_admin_0_countries.zip')

# Merge the world map with the sum co2 based on ISO country codes
world = world.merge(region_emissions, left_on='ADMIN', right_on='Country Name', how='left')

# Filter out rows with count 0
world = world[world['sum'] > 0]

# Create a Folium Map centered at the mean of latitude and longitude
m = folium.Map(location=[world.geometry.centroid.y.mean(), world.geometry.centroid.x.mean()], zoom_start=2)

# Add GeoJSON data to the map
folium.GeoJson(world, name='geojson', tooltip=folium.features.GeoJsonTooltip(fields=['ADMIN', 'sum'], aliases=['Country', 'sum'])).add_to(m)

# Add sum values as text annotations using MarkerCluster
marker_cluster = MarkerCluster().add_to(m)
for idx, row in world.iterrows():
    folium.Marker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        popup=f"{row['ADMIN']}: {row['sum']:.0f}",
    ).add_to(marker_cluster)

# Display the map in the Colab notebook
display(m)

# Correlation Analysis
co2_df = melted_df[["Country Name",'CO2 Emissions (metric tons per capita)']]
gdp_df = world[['Country Name','GDP_MD']]
population_df = world[['Country Name','POP_EST']]

# Merge DataFrames based on a common identifier (e.g., 'Country Name' or 'Country Code')
merged_df = co2_df.merge(gdp_df, on='Country Name')  # Replace with the appropriate merge key
merged_df = merged_df.merge(population_df, on='Country Name')

# Rename columns in the merged DataFrame
merged_df = merged_df.rename(columns={
    'CO2 Emissions (metric tons per capita)': 'CO2 Emissions',
    'GDP_MD': 'GDP',
    'POP_EST': 'Population'
})

# Calculate correlation coefficients
correlations = merged_df[['CO2 Emissions', 'GDP', 'Population']].corr(method='spearman')


# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Scatter Plot
# Scatter plot with regression line for CO2 Emissions vs. GDP
plt.figure(figsize=(12, 6))
sns.regplot(x='GDP', y='CO2 Emissions', data=merged_df, scatter_kws={'s': 50})
plt.title('CO2 Emissions vs. GDP with Regression Line')
plt.xlabel('GDP (Million USD)')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.show()

# Scatter plot with regression line for CO2 Emissions vs. Population
plt.figure(figsize=(12, 6))
sns.regplot(x='Population', y='CO2 Emissions', data=merged_df, scatter_kws={'s': 50})
plt.title('CO2 Emissions vs. Population with Regression Line')
plt.xlabel('Population Estimate')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.show()
