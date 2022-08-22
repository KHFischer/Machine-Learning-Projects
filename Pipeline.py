# Libraries
import numpy as np
import pandas as pd
import datetime

# Feature engineering functions
def unit_incl_tax():
    price = inp['Unit price'] * (1+(inp['Tax percentage']/100))
    return price

def customer_total():
    total = inp['Price incl. Tax'] * inp['Quantity']
    return total
    
def market_total():
    market_total = inp['Unit price'] * inp['Quantity']
    return market_total
    
def unit_profit():
    prof = (inp['Unit price'] * ((inp['gross margin percentage']/100)))
    return prof

def cogs():
    cogs = (inp['Unit price'] - inp['Unit profit']) * inp['Quantity']
    return cogs
    
def profit():
    profit = inp['Market total'] - inp['COGS']
    return profit

def tax_total():
    tax_total = inp['Customer total'] * (1+(inp['Tax percentage']/100))
    return tax_total

def week_num():
    week = pd.DataFrame(pd.DatetimeIndex(inp['Date']).isocalendar().week).reset_index()
    return week['week']
  
# Function that Extracts data, Transforms it using the functions above and Loads the transformed file
def pipeline(path):
    
    # Extract file
    print('Extracting file...')
    
    inp = pd.read_csv(path)
    inp.drop(['Tax 5%', 'Total', 'cogs', 'gross income'], axis=1, inplace=True)
    inp['Tax percentage'] = 5
    inp.sort_values(by='Date', inplace=True)
    
    # Transform file
    print('Transforming file...')
    
    inp['Price incl. Tax'] = unit_incl_tax()
    inp['Customer total'] = customer_total()
    inp['Market total'] = market_total()
    inp['Unit profit'] = unit_profit()
    inp['COGS'] = cogs()
    inp['Profit'] = profit()
    inp['Tax total'] = tax_total()
    inp['Cum. sales'] = inp['Market total'].cumsum()
    inp['Cum. Quantity'] = inp['Quantity'].cumsum()
    inp['Cum. profit'] = inp['Profit'].cumsum()
    inp['Week'] = week_num()
    inp['Month'] = pd.DatetimeIndex(inp['Date']).month_name()
    inp['Year'] = pd.DatetimeIndex(inp['Date']).year
    inp['City'] = inp['City'].replace(['Naypyitaw'], 'Naypyidaw')
    
    # Loading file
    print('Loading file...')
    
    order = ['City', 'Customer type', 'Gender', 'Product line', 'Unit price', 
             'Tax percentage', 'Price incl. Tax', 'gross margin percentage', 
             'Unit profit', 'Quantity', 'Cum. Quantity', 'Customer total', 
             'Market total', 'Cum. sales', 'Tax total','COGS', 'Profit', 
             'Cum. profit', 'Date', 'Week', 'Month', 'Year', 'Time', 'Payment', 'Rating']
    
    ts = pd.Timestamp(datetime.datetime(2022, 12, 19).today())
    
    output = round(inp[order], 2)
    outName = ('output_'+ts.month_name()+'.csv')
    output.to_csv(outName, index=False)
    
    return 'Process complete, file name: '+ outName

# This pipeline was made for the 'Supermarket sales' dataset available on Kaggle
# To run the pipeline and generate transformed data in csv format
pipeline('../input/supermarket-sales/supermarket_sales - Sheet1.csv')

'''
Output file will be named output_CurrentMonth.csv

I made a dashboard for the supermarket using the output file in Power BI, it is available at my personal website!
Thanks for reading!
'''
