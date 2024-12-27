# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace the path if necessary)
df = pd.read_csv('Dataset/UK_Accident.csv', encoding='latin1')

# Display basic dataset information
print("Dataset Head:")
print(df.head())

# Data Preprocessing
# Drop rows with missing values in essential columns
df = df.dropna(subset=['Accident_Severity', 'Weather_Conditions', 'Road_Surface_Conditions', 'Time'])

# Convert 'Time' to just hour (simplified for faster processing)
df['Accident_Hour'] = pd.to_datetime(df['Time'], format='%H%M', errors='coerce').dt.hour

# Label encoding for categorical columns
label_encoder = LabelEncoder()
df['Accident_Severity'] = label_encoder.fit_transform(df['Accident_Severity'])
df['Weather_Conditions'] = label_encoder.fit_transform(df['Weather_Conditions'])
df['Road_Surface_Conditions'] = label_encoder.fit_transform(df['Road_Surface_Conditions'])

# ------------------------------
# 1. Explore Road Conditions, Weather, and Time of Day
# ------------------------------

# Visualizing accident severity by weather conditions (Simplified)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Weather_Conditions', hue='Accident_Severity', palette='Set2')
plt.title('Accident Severity by Weather Conditions')
plt.xticks(rotation=45)
plt.show()

# Visualizing accident severity by road surface conditions
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Road_Surface_Conditions', hue='Accident_Severity', palette='Set1')
plt.title('Accident Severity by Road Surface Conditions')
plt.xticks(rotation=45)
plt.show()

# Visualizing accidents by time of day
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Accident_Hour', hue='Accident_Severity', palette='viridis')
plt.title('Accident Severity by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.show()

# ------------------------------
# 2. Accident Trends (Severity by Weather and Road Surface)
# ------------------------------
# Accident severity count by weather conditions
severity_weather = df.groupby(['Weather_Conditions', 'Accident_Severity']).size().unstack().fillna(0)
severity_weather.plot(kind='bar', stacked=True, figsize=(10, 6), color=sns.color_palette('Set2'))
plt.title('Accident Severity by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

# Accident severity count by road surface conditions
severity_road_surface = df.groupby(['Road_Surface_Conditions', 'Accident_Severity']).size().unstack().fillna(0)
severity_road_surface.plot(kind='bar', stacked=True, figsize=(10, 6), color=sns.color_palette('Set1'))
plt.title('Accident Severity by Road Surface Conditions')
plt.xlabel('Road Surface Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

# ------------------------------
# 3. Correlation Heatmap (for numerical features)
# ------------------------------
# Select numerical columns for correlation
numerical_cols = ['Accident_Severity', 'Weather_Conditions', 'Road_Surface_Conditions', 'Accident_Hour']
correlation = df[numerical_cols].corr()

# Plot a heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
