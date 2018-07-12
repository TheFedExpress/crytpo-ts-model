# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:40:24 2018
@author: Peter_goodridge
"""

def get_current(ticker, trade_type):
    import requests
    
    url = 'https://bittrex.com/api/v1.1/public/getticker?market=USDT-{}'.format(ticker.upper())
    response = requests.get(url)
    json_text = response.json()['result']

    bid = float(json_text['Bid'])
    ask = float(json_text['Ask'])
    last = float(json_text['Last'])
    
    if trade_type in ('buy', 'cover'):
        return ask
    elif trade_type in ('sell', 'short'):
        return bid
    elif trade_type == 'check':
        return(last)
        
def get_history(ticker, interval, frequency, rows, exchange = ''):
    import requests
    import pandas as pd
    from datetime import datetime
    
    ticker = ticker.upper()
    
    if exchange != '':
       exchange =  '&e=' + exchange
    
    url = 'https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym=USD&limit={}&aggregate={}{}'.format(
            interval, ticker, rows, frequency, exchange)
    response = requests.get(url)
    obj = response.json()['Data']
    
    df = pd.DataFrame(obj, columns = ['time', 'close', 'high', 'low', 'open', 'volumefrom', 'volumeto'])
    df['time'] = df['time'].map(lambda x: datetime.fromtimestamp(x))
    df = df[df['close'] != 0]
    df.sort_values(by = ['time'], inplace = True)
    df.set_index('time', inplace = True)
    return df

def get_sd(ticker, days):
    from datetime import datetime, timedelta
    import requests
    import pandas as pd
    
    end = datetime.now()
    
    start = (end.date() - timedelta(days = days))
    url = 'https://api.gdax.com/products/BTC-USD/candles?start={}&end={}&granularity=86400'.format(start,end)
    print (url)
    obj = requests.get(url).json()
    print(obj)
    df = pd.DataFrame(obj, columns = ['time', 'low', 'high', 'open', 'close', 'volume'])
    return df

def get_24(ticker):
    import requests
    import pandas as pd
    
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym=USDT'.format(ticker.upper())
    response = requests.get(url)
    obj = response.json()['Data']
    
    df = pd.DataFrame(obj, columns = ['time', 'low', 'high', 'open', 'close', 'volume'])
    raw = tuple(df['close'].describe().iloc[[1,2,3,7]])

    formated = tuple(['${:,.2f}'.format(item) for item in raw])
    return formated
    

def make_chart(ticker):
    from plotly.graph_objs import Scatter, Data, Line
    import requests
    from datetime import datetime
    import pandas as pd
    
    ticker = ticker.upper()
    url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym=USDT&limit=119&aggregate=1'.format(ticker)
    response = requests.get(url)
    obj = response.json()['Data']
    
    df = pd.DataFrame(obj, columns = ['time', 'low', 'high', 'open', 'close', 'volume'])
    df.sort_values(by = ['time'], inplace = True)
    dates = df['time'].map(datetime.fromtimestamp)
    df.loc[:, 'time'] = dates
    df.loc[:, 'moving'] = df['close'].rolling(window = 20).mean()
    
    labels = dates.map('{:%Y-%m-%d}'.format)[19:].values
    
    price = Scatter(x= labels, y = df.loc[19:, 'close'], line = Line(width = 2, color = 'blue'), name = ticker)
    moving = Scatter(x= labels, y = df.loc[19:, 'moving'], line = Line(width = 2, color = 'orange'), name = '20 Day Moving Avg')
    data = Data([price, moving])
    return data

def find_actives():
    import requests
    import pandas as pd
    
    url = 'https://bittrex.com/api/v1.1/public/getmarkets'
    response = requests.get(url)
    obj = response.json()['result']
    
    df = pd.DataFrame(obj)
    actives = list(df[df.BaseCurrency == 'USDT']['MarketCurrency'].values)
    return sorted(actives)