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

