📈 Stock Market Predictor
A Streamlit web app for predicting stock prices using LSTM deep learning models.

📝 Overview
This project uses Long Short-Term Memory (LSTM) networks to predict stock prices based on historical data. It is built with:
✅ Streamlit (Web App UI)
✅ Yahoo Finance API (Stock Data)
✅ Keras & TensorFlow (LSTM Model)
✅ Scikit-Learn (Data Preprocessing)
✅ Matplotlib (Data Visualization)

🎯 Features
✅ Fetch real-time stock data using Yahoo Finance API
✅ Train an LSTM model on stock data
✅ Predict future stock prices based on past trends
✅ Compare Moving Averages (50, 100 days) with real stock prices
✅ Interactive Streamlit UI for stock selection and visualization

📌 Installation
1️⃣ Clone the Repository

git clone https://github.com/yourusername/Stock-Predictor.git
cd Stock-Predictor

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py
📊 Model Overview
Uses LSTM layers to learn stock price trends.

Trained on 80% of stock data and tested on 20%.

MinMaxScaler normalizes the data for better model performance.

Predicts stock prices based on the last 100 days of data.

⚙️ Technologies Used
Python

Streamlit

TensorFlow & Keras

Scikit-Learn

Pandas & NumPy

Matplotlib

Yahoo Finance API

