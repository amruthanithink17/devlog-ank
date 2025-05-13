'''This Python script is a sales data analysis pipeline that uses
pandas, matplotlib, seaborn, and statsmodels to load, process,
and visualize time series sales data from a CSV file. '''

'''1. Set Sample Dataset Structure (CSV)
This step is a comment placeholder. The actual dataset is assumed to have columns like:

1.Date: Date of transaction

2.Product: Product name

3.Region: Region of sales

4.Units_Sold: Quantity sold

5.Unit_Price: Price per unit'''

# 1.set sample dataset structure(CSV)
# 2.Required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime

# 3.load and prepare data
#load data
#df=pd.read_csv('sales_data.csv',parse_dates=['Date'])
df = pd.read_csv('C:/Users/acer/Desktop/NA/sales_data.csv', parse_dates=['Date'], date_format='%d-%m-%y')



#create 'sales' if not already there
if 'Sales' not in df.columns:
    df['Sales']=df['Units_Sold']*df['Unit_Price']

#set date as index
df.set_index('Date',inplace=True)

#optional:fill missing values
df.fillna(0,inplace=True)

# 4.sales over time
#monthly sales
monthly_sales=df['Sales'].resample('ME').sum()
plt.figure(figsize=(12,6))
monthly_sales.plot()
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid()
plt.show()

# 5.time series decomposition
'''seasonal_decompose() needs at least 2 full seasonal
cycles to work correctly. For monthly data, that means:
You need 24 months of data (2 years)'''
#But you only have 12 months (1 year of 2024)

'''decomposed = seasonal_decompose(monthly_sales, model='additive')
decomposed.plot()
plt.show()'''

# 6.Top Products and Regions
# Total Sales by Product
top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
top_products.plot(kind='bar', figsize=(10, 5), title='Total Sales by Product')
plt.ylabel('Sales')
plt.show()

# Sales by Region
region_sales = df.groupby('Region')['Sales'].sum()
region_sales.plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6), title='Sales by Region')
plt.ylabel('')
plt.show()

# 7.Monthly Growth Rate
growth = monthly_sales.pct_change().fillna(0) * 100
plt.figure(figsize=(12, 5))
growth.plot(kind='bar')
plt.title('Monthly Sales Growth Rate (%)')
plt.ylabel('Growth %')
plt.grid()
plt.show()

# 8.optional
# Heatmap of correlation if additional data exists
correlation = df[['Units_Sold', 'Unit_Price', 'Sales']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

'''Final Output Summary
1.Line graph of monthly sales trend

2.Seasonality decomposition(requires 2+ years of data)

3.Bar charts of top products and growth rate

4.Pie chart of region-wise performance

5.Heatmap -Correlation between sales drivers (optional)'''




