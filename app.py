import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

# Load pre-trained model
model = load_model(r'C:\Users\Acer\Desktop\stock\Stock predicction model.keras')

# Streamlit UI
st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'TSLA')
start = '2015-01-01'
end = '2025-01-01'

# Fetch Stock Data
data = yf.download(stock, start, end)

# Display Data
st.subheader('Stock Data')
st.write(data)

# Split Data into Training and Testing
data_train = pd.DataFrame(data.Close[:int(len(data) * 0.80)])  # First 80% for training
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])   # Last 20% for testing

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Preparing Test Data
past_100_days = data_train.tail(100)  # Use last 100 days from training
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1=  plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2=  plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2)



# Creating Input Sequences
x, y = [], []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])  # 100 previous days as features
    y.append(data_test_scaled[i, 0])     # Next day's price as label

x, y = np.array(x), np.array(y)
y = y.reshape(-1, 1)





# Make Predictions
predict = model.predict(x)

# Correct Scaling
scale = 1 / scaler.scale_[0]  
predict = predict * scale
y = y * scale

st.subheader('Original Price vs  Predicted Price')
ma_100_days = data.Close.rolling(100).mean()
fig3=  plt.figure(figsize=(8,6))
plt.plot(predict,'r',label='Original Price')
plt.plot(y,'g',label=' Predicted Price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig3)
