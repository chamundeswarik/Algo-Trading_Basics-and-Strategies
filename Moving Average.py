
#%% 
import pandas as pd
import numpy as np
import datetime #as we deal with time series data
import matplotlib.pyplot as plt # to plot graphs and complex statistical/financial analysis
import os #operating system based library inorder to change our working directory, shift within python console
import pandas_datareader.data as web

end=datetime.datetime.now().date() # in datetime library, function datetimenow is including, so choosing date
start=end-pd.Timedelta(days=365*10) # so we have given start point and end point i.e. from 10 years back
df=web.DataReader("AAPL","quandl",start,end) #quandl API works and it has quite bit data, it also gives a lot of access with their library- try downloading their library, needs a free account
df=df.reindex(index=df.index[::-1]) #reversing the order
df.to_csv('AAPL.csv') #we can save our downloaded data to csv



#%%
# we buy when price>10-month SMA, Sell and move to cash when price<10-month SMA
buyPrice = 0.0
sellPrice = 0.0
cash = 1
stock = 0
sma = 200
maWealth = 1.0

ma= np.round(df['AdjClose'].rolling(window=sma, center=False).mean(), 2) #getting rounded to 2 places
n_days = len(df['AdjClose']) #length of AdjClose
closePrices = df['AdjClose']

#initiaised some variables here
buy_data = []
sell_data = []
trade_price = []
wealth = []
for d in range(sma-1, n_days):
    # buy if Stockprice > MA & if not bought yet
    if closePrices[d] > ma[d] and cash == 1:
        buyPrice = closePrices[d + 1]
     #buying at tomorrow closing price once u see Close price of today > 10-month SMA today
        buy_data.append(buyPrice)
        trade_price.append(buyPrice)
        cash = 0
        stock = 1
   
    # sell if Stockprice < MA and if you have a stock to sell
    if closePrices[d] < ma[d]  and stock == 1:
        sellPrice = closePrices[d + 1]
        cash = 1
        stock = 0
        sell_data.append(sellPrice)
        trade_price.append(sellPrice)
                 maWealth=maWealth*(sellPrice/buyPrice)
         wealth.append(maWealth)
    
    tp=pd.DataFrame(trade_Price)
    
