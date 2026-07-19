#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

price = pd.read_csv("forex_price_data.csv")

trade = pd.read_csv("trade_log.csv")

mc = pd.read_csv("mc_scenarios.csv")


# In[2]:


price.head()
trade.head()
mc.head()


# In[3]:


price.info()

trade.info()

mc.info()


# In[4]:


mc.isnull().sum()


# In[5]:


trade.isnull().sum()


# In[6]:


price.isnull().sum()


# In[7]:


price['DateTime'] = pd.to_datetime(price['DateTime'])


# In[8]:


trade['EntryDateTime'] = pd.to_datetime(trade['EntryDateTime'])

trade['ExitDateTime'] = pd.to_datetime(trade['ExitDateTime'])


# In[9]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(price["DateTime"],price["Close"])
plt.title("Closing Price")
plt.show()


# In[10]:


plt.figure(figsize=(10,5))
plt.bar(price["DateTime"],price["Volume"])
plt.show()


# In[11]:


plt.figure(figsize=(10,5))
plt.plot(price["DateTime"],price["Spread"])
plt.show()


# In[12]:


price['SMA10'] = price['Close'].rolling(window=10).mean()

price['SMA20'] = price['Close'].rolling(window=20).mean()


# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(price['DateTime'], price['Close'], label='Close')
plt.plot(price['DateTime'], price['SMA10'], label='SMA10')
plt.plot(price['DateTime'], price['SMA20'], label='SMA20')
plt.legend()
plt.title("Simple Moving Average")
plt.show()


# In[14]:


price['EMA10'] = price['Close'].ewm(span=10, adjust=False).mean()

price['EMA20'] = price['Close'].ewm(span=20, adjust=False).mean()


# In[15]:


plt.figure(figsize=(12,6))
plt.plot(price['DateTime'], price['Close'], label='Close')
plt.plot(price['DateTime'], price['EMA10'], label='EMA10')
plt.plot(price['DateTime'], price['EMA20'], label='EMA20')
plt.legend()
plt.title("Exponential Moving Average")
plt.show()


# In[16]:


price[['Close','SMA10','EMA10']].tail(10)


# In[17]:


price['SMA5'] = price['Close'].rolling(5).mean()
price['EMA5'] = price['Close'].ewm(span=5, adjust=False).mean()


# In[18]:


plt.figure(figsize=(12,6))
plt.plot(price['DateTime'], price['Close'], label='Close')
plt.plot(price['DateTime'], price['SMA5'], label='SMA5')
plt.plot(price['DateTime'], price['EMA5'], label='EMA5')
plt.legend()
plt.show()


# In[19]:


pip install ta


# In[20]:


from ta.momentum import RSIIndicator

price['RSI'] = RSIIndicator(close=price['Close'], window=14).rsi()


# In[21]:


plt.figure(figsize=(12,5))
plt.plot(price['DateTime'], price['RSI'])
plt.axhline(70, color='red')
plt.axhline(30, color='green')
plt.title("RSI")
plt.show()


# In[22]:


from ta.trend import MACD

macd = MACD(close=price['Close'])

price['MACD'] = macd.macd()
price['Signal_Line'] = macd.macd_signal()


# In[23]:


plt.figure(figsize=(12,6))
plt.plot(price['DateTime'], price['MACD'], label='MACD')
plt.plot(price['DateTime'], price['Signal_Line'], label='Signal')
plt.legend()
plt.title("MACD")
plt.show()


# In[24]:


from ta.volatility import BollingerBands

bb = BollingerBands(close=price['Close'])

price['UpperBand'] = bb.bollinger_hband()
price['MiddleBand'] = bb.bollinger_mavg()
price['LowerBand'] = bb.bollinger_lband()


# In[25]:


plt.figure(figsize=(12,6))

plt.plot(price['DateTime'], price['Close'], label='Close')
plt.plot(price['DateTime'], price['UpperBand'], label='Upper')
plt.plot(price['DateTime'], price['MiddleBand'], label='Middle')
plt.plot(price['DateTime'], price['LowerBand'], label='Lower')

plt.legend()
plt.title("Bollinger Bands")
plt.show()


# In[26]:


price.to_csv("forex_price_processed.csv", index=False)


# In[27]:


price[['SMA5','EMA5','RSI','MACD','Signal_Line',
       'UpperBand','MiddleBand','LowerBand']].isnull().sum()


# In[28]:


price = price.dropna().reset_index(drop=True)


# In[29]:


price.isnull().sum()


# In[30]:


price['Signal'] = 'HOLD'

buy_condition = (price['RSI'] < 30) & (price['MACD'] > price['Signal_Line'])
sell_condition = (price['RSI'] > 70) & (price['MACD'] < price['Signal_Line'])

price.loc[buy_condition, 'Signal'] = 'BUY'
price.loc[sell_condition, 'Signal'] = 'SELL'


# In[31]:


price['Signal'].value_counts()


# In[32]:


import matplotlib.pyplot as plt

price['Signal'].value_counts().plot(kind='bar')
plt.title("Trading Signals")
plt.xlabel("Signal")
plt.ylabel("Count")
plt.show()


# In[33]:


price.to_csv("forex_price_processed.csv", index=False)


# In[34]:


trade.head()
trade.info()
trade.describe()


# In[35]:


trade['PnL'].sum()


# In[36]:


trade['PnL'].mean()


# In[37]:


trade.groupby('PairName')['PnL'].sum().plot(kind='bar')

plt.title("Profit by Currency Pair")
plt.show()


# In[38]:


trade.groupby('StrategyName')['PnL'].sum().plot(kind='bar')

plt.title("Profit by Strategy")
plt.show()


# In[39]:


trade['Direction'].value_counts().plot(kind='pie',autopct='%1.1f%%')

plt.title("Trade Direction")
plt.show()


# In[40]:


trade['HoldingDays'].hist()

plt.title("Holding Days Distribution")
plt.show()


# In[41]:


trade['PnL'].hist()

plt.title("Profit and Loss Distribution")
plt.show()


# In[42]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


# In[43]:


trade = pd.read_csv("trade_log.csv")


# In[44]:


trade['SignalType'].value_counts()


# In[45]:


encoder = LabelEncoder()

trade['SignalType'] = encoder.fit_transform(trade['SignalType'])


# In[46]:


X = trade[['EntryPrice',
           'ExitPrice',
           'PositionSize',
           'HoldingDays',
           'Commission']]


# In[47]:


y = trade['SignalType']


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# In[49]:


rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

prediction = rf.predict(X_test)


# In[50]:


print("Accuracy")

print(accuracy_score(y_test, prediction))

print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))


# In[51]:


xgb = XGBClassifier()

xgb.fit(X_train, y_train)

prediction = xgb.predict(X_test)


# In[52]:


print(accuracy_score(y_test, prediction))


# In[53]:


nn = MLPClassifier(hidden_layer_sizes=(100,),
                   max_iter=500,
                   random_state=42)

nn.fit(X_train, y_train)

prediction = nn.predict(X_test)


# In[54]:


print(accuracy_score(y_test, prediction))

print(classification_report(y_test, prediction))


# In[55]:


trade['SignalType'].value_counts()


# In[56]:


from sklearn.metrics import accuracy_score

# Random Forest
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# XGBoost
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

# Neural Network
nn_pred = nn.predict(X_test)
nn_acc = accuracy_score(y_test, nn_pred)


# In[57]:


import pandas as pd

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
    'Accuracy': [rf_acc, xgb_acc, nn_acc]
})

print(comparison)


# In[58]:


import pandas as pd
import matplotlib.pyplot as plt

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
    'Accuracy': [0.26, 0.30, 0.16]
})

print(comparison)

plt.figure(figsize=(8,5))
plt.bar(comparison['Model'], comparison['Accuracy'])

plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

for i, v in enumerate(comparison['Accuracy']):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.show()


# In[59]:


trade['SignalType'].unique()


# In[60]:


trade[['SignalType', 'Direction', 'StrategyName']].head(10)


# In[61]:


import pandas as pd

mc = pd.read_csv("mc_scenarios.csv")


# In[62]:


print("Average Portfolio PnL:", mc['PortfolioPnL'].mean())

print("Maximum Portfolio PnL:", mc['PortfolioPnL'].max())

print("Minimum Portfolio PnL:", mc['PortfolioPnL'].min())


# In[63]:


import numpy as np

var95 = np.percentile(mc['PortfolioPnL'], 5)

print("Value at Risk (95%):", var95)


# In[64]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.hist(mc['PortfolioPnL'], bins=30)

plt.title("Portfolio PnL Distribution")
plt.xlabel("Portfolio PnL")
plt.ylabel("Frequency")

plt.show()


# In[65]:


plt.figure(figsize=(8,5))

plt.scatter(mc['ParallelShift'], mc['PortfolioPnL'])

plt.title("Parallel Shift vs Portfolio PnL")
plt.xlabel("Parallel Shift")
plt.ylabel("Portfolio PnL")

plt.show()


# In[66]:


plt.figure(figsize=(8,5))

plt.scatter(mc['CorrShock'], mc['PortfolioPnL'])

plt.title("Correlation Shock vs Portfolio PnL")
plt.xlabel("Correlation Shock")
plt.ylabel("Portfolio PnL")

plt.show()


# In[67]:


plt.figure(figsize=(8,5))

plt.scatter(mc['VolShock'], mc['PortfolioPnL'])

plt.title("Volatility Shock vs Portfolio PnL")
plt.xlabel("Volatility Shock")
plt.ylabel("Portfolio PnL")

plt.show()


# In[68]:


mc.to_csv("mc_processed.csv", index=False)


# In[69]:


trade.to_csv("trade_processed.csv", index=False)


# In[1]:


import pandas as pd

price = pd.read_csv("forex_price_processed.csv")


# In[2]:


price.columns


# In[4]:


price['Future_Close'] = price['Close'].shift(-1)

price['Target'] = (price['Future_Close'] > price['Close']).astype(int)


# In[5]:


price = price.dropna().reset_index(drop=True)


# In[6]:


price.isnull().sum()


# In[7]:


X = price[
    [
        'SMA10',
        'EMA10',
        'RSI',
        'MACD',
        'Signal_Line',
        'UpperBand',
        'MiddleBand',
        'LowerBand'
    ]
]

y = price['Target']


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)


# In[10]:


from xgboost import XGBClassifier

xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)

print("XGBoost Accuracy:", xgb_acc)


# In[11]:


from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(
    hidden_layer_sizes=(64,32),
    max_iter=1000,
    random_state=42
)

nn.fit(X_train, y_train)

nn_pred = nn.predict(X_test)

nn_acc = accuracy_score(y_test, nn_pred)

print("Neural Network Accuracy:", nn_acc)


# In[12]:


import pandas as pd

comparison = pd.DataFrame({
    'Model': [
        'Random Forest',
        'XGBoost',
        'Neural Network'
    ],
    'Accuracy': [
        rf_acc,
        xgb_acc,
        nn_acc
    ]
})

print(comparison)


# In[13]:


price.to_csv("forex_price_processed.csv", index=False)


# In[14]:


price['PairName'].value_counts()


# In[15]:


forex = pd.read_csv("forex_price_data.csv")
forex['PairName'].value_counts()


# In[16]:


price['PairName'].value_counts()


# In[17]:


forex['PairName'].value_counts()


# In[19]:


import pandas as pd

trade = pd.read_csv("trade_processed.csv")


# In[20]:


trade["SignalType"].value_counts()


# In[21]:


signal_map = {
    0: "BUY",
    1: "SELL",
    2: "STRONG_BUY",
    3: "STRONG_SELL"
}

trade["SignalType"] = trade["SignalType"].replace(signal_map)


# In[22]:


trade["SignalType"].value_counts()


# In[23]:


trade.to_csv("trade_processed.csv", index=False)


# In[24]:


print(trade["SignalType"].dtype)

print(trade["SignalType"].unique())


# In[25]:


trade = pd.read_csv("trade_processed.csv")
print(trade["SignalType"].head())


# In[1]:


import pandas as pd
import numpy as np

trade = pd.read_csv("trade_processed.csv")

# Calculate return per trade
trade["Return"] = trade["PnL"] / trade["PositionSize"]

# Sharpe Ratio
sharpe = trade["Return"].mean() / trade["Return"].std()

print("Sharpe Ratio =", sharpe)


# In[2]:


annual_sharpe = sharpe * np.sqrt(252)

print("Annualized Sharpe =", annual_sharpe)


# In[3]:


print("Mean Return:", trade["Return"].mean())
print("Std Return:", trade["Return"].std())


# In[4]:


import pandas as pd

trade = pd.read_csv("trade_processed.csv")

trade["Return"] = trade["PnL"] / trade["PositionSize"]

trade.to_csv("trade_processed.csv", index=False)

print(trade.head())


# In[2]:


get_ipython().system('jupyter nbconvert --to script "Forex(1).ipynb"')


# In[ ]:




