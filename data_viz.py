#!/usr/bin/env python3

"""
data_viz.py

Description:
This script reads data from a CSV file and creates a scatter plot using matplotlib
and seaborn libraries to visualize the data.

Dependencies:
- pandas
- matplotlib
- seaborn

Usage:
Ensure you have installed the required dependencies using pip:
    pip install pandas matplotlib seaborn

Then, run the script with the desired data file:
    ./data_viz.py data.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    if len(sys.argv) != 2:
        print("Usage: ./data_viz.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]

    # Read data from a CSV file
    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{data_file}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Unable to parse '{data_file}'. Please ensure it's a valid CSV file.")
        sys.exit(1)

    # Display the first 5 rows of the DataFrame
    print("First 5 rows of the DataFrame:")
    print(data.head())

    # Assuming 'x' and 'y' are columns in your dataset
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='Velocity', y='Value')
    plt.xlabel('Velocity')
    plt.ylabel('Value')
    plt.title('Scatter Plot')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

