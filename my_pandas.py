#!/usr/bin/python3
import pandas as pd

# Read data from a CSV file
data = pd.read_csv('data.csv')

# Display the first 5 rows of the DataFrame
print(data.head())

# Filter rows based on a condition
filtered_data = data[data['Velocity'] > 100]

# Handle missing values
data.dropna(inplace=True)

