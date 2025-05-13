# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from textblob import TextBlob

# Sample dataset (you can replace this with your actual dataset)
data = {
    'Date': ['2023-03-01', '2023-03-02', '2023-03-03', '2023-03-04', '2023-03-05'],
    'Post Type': ['Image', 'Video', 'Text', 'Image', 'Video'],
    'Likes': [150, 200, 100, 120, 250],
    'Comments': [20, 35, 10, 15, 50],
    'Shares': [10, 25, 5, 8, 30]
}

# Create DataFrame
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])  # Convert Date to datetime format

# --- Step 1: Visualize Engagement Trends Over Time ---
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Likes'], label='Likes', marker='o')
plt.plot(df['Date'], df['Comments'], label='Comments', marker='x')
plt.plot(df['Date'], df['Shares'], label='Shares', marker='^')

plt.title('Social Media Engagement Over Time')
plt.xlabel('Date')
plt.ylabel('Engagement Count')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Step 2: Content Type Performance ---
# Group by 'Post Type' and calculate the average engagement metrics
content_performance = df.groupby('Post Type')[['Likes', 'Comments', 'Shares']].mean()

# Bar chart to compare average engagement per content type
content_performance.plot(kind='bar', figsize=(10, 6))
plt.title('Average Engagement by Post Type')
plt.ylabel('Average Engagement Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Step 3: Engagement Correlation ---
# Calculate correlation matrix for Likes, Comments, and Shares
correlation_matrix = df[['Likes', 'Comments', 'Shares']].corr()

# Plot a heatmap to show the correlation between the engagement metrics
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Engagement Correlation Heatmap')
plt.show()

# --- Step 4: Predicting Future Engagement (Linear Regression for Likes) ---
# Convert 'Date' to a numerical value (days since the first date)
df['Date_num'] = (df['Date'] - df['Date'].min()).dt.days

# Features and target variable
X = df[['Date_num']]  # Feature: Date in numerical form
y = df['Likes']  # Target: Likes

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the number of likes for the next day
next_day = pd.DataFrame({'Date_num': [df['Date_num'].max() + 1]})  # Next day's numerical date as a DataFrame
predicted_likes = model.predict(next_day)

print(f"Predicted likes for the next day: {predicted_likes[0]:.2f}")

# --- Step 5: Sentiment Analysis on Comments (Optional) ---
# Sample comment data
comments = ['Love this post!', 'Not a fan of this content', 'Great video, very informative!']

# Perform sentiment analysis on comments
sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
print(f"Sentiment analysis of comments: {sentiments}")

# --- Optional: Save the results ---
# You can save the DataFrame or model predictions to a file for later use
df.to_csv('social_media_engagement.csv', index=False)
print("Data saved to 'social_media_engagement.csv'")
