
#%%
import pandas as pd # import pandas
dir(pd) # to know the list of commands available
pd.__version__ # to know the version
df=pd.DataFrame() # initialise dataframe
data=pd.read_csv('MSFT.csv') # saved the data in the default locataion and read csv, size on right variable window gives (rows, columns)
data2=pd.read_csv('MSFT.csv', index_col='Date') # removed the numbering column and giving date itself index
data3=data2.copy() # not to tamper original data and replicating to edit
data2.shape #dimensions of dataframe, rows and columns, it appears as tuple
data2.info() #to check available information
data2.head(2) #to get the top number of rows, if nothing given by default they give first 5 / last 5 - need to specify the number if any specific number required
data2.tail(1) # to get bottom row (in both header doesn't count as a separate row item)
data2.index #gives that date is index here and gives all the dates
data2.columns
data2.values
data.index # here it will be different to data2.index
#indexing is pulling out subsets of data to pull certain rows or columns [], .loc[],.iloc[]
#[]
Open = data2['Open'] # when u give equal to and assign a new name , it comes as a separate series in variable explorer
Close = data2.Open # here we can give the brackets if u have space in name, as dot operator doesn't wrk with space
close_close=data2[['Open']] #double brackets give as dataframe instead of series- as series has only one dimension and dataframe has two
data2['New'] =0 #new column New gets created with value 0 for all rows

#.loc[]
#if I want to select single row instead of column
data2.loc['2011-01-07'] #gives the entire row information 
data2.loc[['2011-01-07','2011-01-14']] #can select multiple rows by giving list in double brackets
data2.loc['2011-01-07':'2011-01-30'] #use single bracket and colon to select range of rows
data2.loc['2011-01-07':'2011-01-30':3] #select every alternate 3rd row
data2.loc['2011-01-07':'2011-01-14',['Open','High']] # to select separate row and columns
data2.loc[['2011-01-07','2011-01-14'],['Open','High']]# always for range no bracket and for specific row selections brackets need to be there

#.iloc[] - integer location indexer
#as pandas index starts with 0
data2.iloc[4,1]
data2.iloc[[4,1]] # u can try various combinations of slices of rows, columns etc.
data2.iat[4,1] #.iat, .iloc are similar
data2['cc']=100*data2['Adj Close'].pct_change() #to calculate returns based on Close
data2['co']=100*(data2['Open']/data2['Close'].shift(1)-1) # shift(1) shifts the column left to it by one unit down
# hence here it's today open/yesterday close
data2['co']=100*(data2['Open']/data2['Close'] -1)
# this is today open/today close

#moving average - .rolling function - rolling is something is getting calculated, here it's mean, window goes backward
n=50 #used window as 50
data2['MA']=data2['Close'].rolling(window=n,center=False).mean()

'''data reader helps download data automatically,
go to anaconda prompt - pip install pandas-datareader
to check: pip show pandas-datareader
if any issues with install, run the below- which upgrades pip package manager and then u can go and install pandas datareader:
    python -m pip install --upgrade pip'''

#%%
import pandas as pd
import numpy as np
import datetime #as we deal with time series data
import matplotlib.pyplot as plt # to plot graphs and complex statistical/financial analysis
import os #operating system based library inorder to change our working directory, shift within python console
import pandas_datareader.data as web
os.getcwd() #gives the current directory
data=pd.read_csv('MSFT.csv')
data2=pd.read_csv('MSFT.csv',index_col='Date',parse_dates=True) #parse_dates ensures we have datetime index i.e. with timestamp
data2.loc['May 2014'] #as it's saved as parse_dates it recognizes this format

end=datetime.datetime.now().date() # in datetime library, function datetime, now is including, so choosing date
start=end-pd.Timedelta(days=365*10) # so we have given start point and end point i.e. from 10 years back
df=web.DataReader("AAPL","quandl",start,end) #quandl API works and it has quite bit data, it also gives a lot of access with their library- try downloading their library, needs a free account
df=df.reindex(index=df.index[::-1]) #reversing the order
df.to_csv('AAPL.csv') #we can save our downloaded data to csv
df.plot() #to plot the graph
df[['Open','Close']].plot()
df[['Open','Close']].plot(grid=True, linewidth=0.5, figsize=(8,5)) #to give more explanation with grids and etc.
df[['Close']].plot(linewidth=0.5)

#%%
# we buy when price>10-month SMA, Sell and move to cash when price<10-month SMA
buyPrice = 0.0
sellPrice = 0.0
maWealth = 1.0
cash = 1
stock = 0
sma = 200

ma= np.round(df['AdjClose'].rolling(window=sma, center=False).mean(), 2) #getting rounded to 2 places
n_days = len(df['AdjClose']) #length of AdjClose
closePrices = df['AdjClose']

buy_data = []
sell_data = []
trade_price = []
wealth = []

for d in range(sma-1,n_days):
    #buy if stock price>MA & if not bought yet
    if closePrices[d] > ma[d] and cash==1;:
        buyPrice = closePrices[d+1]
        
        buy_data.append(buyPrice)
        trade_price.append(buyPrice)
        cash=0
        stock=1
      #sell if stockprice<MA and if you have stock to sell
     if closePrices[d]<ma[d] and stock ==1:
         sellPrice=closePrices[d+1]
         
         cash=1
         stock=0
         sell_data.append(sellPrice)
         trade_price.append(sellPrice)
         maWealth=maWealth*(sellPrice/buyPrice)
         wealth.append(maWealth)
    
    tp=pd.DataFrame(trade_Price)
#%%

import pickle #one more package to save objects
#Moving average crossover strategy
m=50 #short lookback period
n=150 #longer lookback period
#buy when SMA>LMA, LMA<SMA on previous day
#sell when SMA<LMA, LMA>SMA on previous day
df['ShortMA']=df['AdjClose'].rolling(window=m, center=False).mean()
df['LongMA']=df['AdjClose'].rolling(window=n, center=False).mean()
#calculating previous day shortMA and longMA- so used .shift(1)
df[['AdjClose','ShortMA','LongMA']].plot(grid=True,linewidth=0.5)
df['ShortMA2']=df['AdjClose'].rolling(window=m,center=False).mean().shift(1)
df['LongMA2']=df['AdjClose'].rolling(window=n,center=False).mean().shift(1)

# df2 = df2.iloc[n-1:] np.where is if condition is true a else b (similar to if condition)

df['Signal'] = np.where((df['ShortMA'] > df['LongMA']) 
                        & (df['ShortMA2'] < df['LongMA2']), 1, 0) #these are vectorised operations
df['Signal'] = np.where((df['ShortMA'] < df['LongMA']) 
                        & (df['ShortMA2'] > df['LongMA2']), -1, df['Signal']) #here if not true u want to remain the signal what it was before
#.apply to perform operations row wise - use axis=1 to make sure the condition gets applied to every single row
#lambda function is anonymous function
df['Buy'] = df.apply(lambda x : x['AdjClose'] if x['ShortMA'] > x['LongMA'] 
                        and x['ShortMA2'] < x['LongMA2'] else 0, axis=1) 

df['Sell'] = df.apply(lambda y : -y['AdjClose'] if y['ShortMA'] < y['LongMA'] 
                        and y['ShortMA2'] > y['LongMA2'] else 0, axis=1)

df['TP'] = df['Buy'] + df['Sell'] #after you calculate sum of buy and sell
df['TP']=df['TP'].replace(to_replace=0, method='ffill') #here I want to maintain TP price by understanding whether i have taken position/not, so replacing all my zeros with my previously traded price - ffil:forward fill, bfil-backward fill

df['Position'] = df['Signal'].replace(to_replace=0, method= 'ffill')
k = df['TP'].nonzero() #taking out all nonzero entries to see the trading prices

type(k) # this tells us k is a tuple

k[0] #this gives us a numpy array
type(k[0]) #confirms that this is a numpy ndarray
len(k[0]) # total number of positions
frame = df.iloc[k]

df['Signal'].value_counts()
# In the period, we've chosen, this is in line with what we see in k

df['Position'].plot(linewidth=1)

#%%
# Alternate way to plot the graph
plt.figure(figsize=(10, 5))
plt.plot(df['Position'])
plt.legend() # not working don't understand why
plt.title("Signal showing buying/selling positions")
plt.xlabel('Time')
plt.tight_layout()
plt.show() # will display the current figure that you are working on
# needed when you run on the interactive console

# Now testing the efficacy of the strategy in comparison to a Buy & Hold approach
# We will use log returns this time

df['Buy & Hold Returns'] = np.log(df['AdjClose'] / df['AdjClose'].shift(1))
df['Strategy Returns'] = df['Buy & Hold Returns'] * df['Position'].shift(1)

df[['Buy & Hold Returns', 'Strategy Returns']].cumsum().plot(grid=True, figsize=(9,5))


# for computing Exponential Moving Average
data2['EMA'] = data2['Adj Close'].ewm(span=40, min_periods=40, freq='D').mean()

#%%
from concurrent import futures
#provides high level API to download 'n' number of data parallely instead of serially



















