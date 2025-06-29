import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

st.title("株価予測モデル（線形回帰） - 2年学習＋直近30日予測")

ticker = st.text_input("銘柄コードを入力してください（例: 5411.T）", value="5411.T")

if ticker:
    # データ取得
    data = yf.download(ticker, period="2y", auto_adjust=True).dropna()
    
    # 安全に1次元Seriesに変換
    close = pd.Series(data['Close'].values.flatten(), index=data.index)
    
    # テクニカル指標計算
    data['SMA_20'] = close.rolling(window=20).mean()
    data['SMA_50'] = close.rolling(window=50).mean()
    data['RSI_14'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    
    data.dropna(inplace=True)
    
    # 翌日の終値を目的変数に（dataのCloseを使う）
    data['Close_next'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    
    # 学習データとテストデータを分割（直近30日をテスト）
    test_size = 30
    train_data = data.iloc[:-test_size]
    test_data = data.iloc[-test_size:]
    
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal']
    X_train = train_data[features]
    y_train = train_data['Close_next']
    X_test = test_data[features]
    y_test = test_data['Close_next']
    
    # モデル学習
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 予測
    train_data['Predicted_Close'] = model.predict(X_train)
    test_data['Predicted_Close'] = model.predict(X_test)
    
    # 精度評価
    rmse = np.sqrt(mean_squared_error(y_test, test_data['Predicted_Close']))
    mae = mean_absolute_error(y_test, test_data['Predicted_Close'])
    mape = np.mean(np.abs((y_test - test_data['Predicted_Close']) / y_test)) * 100
    
    st.write(f"直近30日予測の精度指標")
    st.write(f"- RMSE: {rmse:.2f}")
    st.write(f"- MAE: {mae:.2f}")
    st.write(f"- MAPE: {mape:.2f} %")
    
    # グラフ描画
    plt.figure(figsize=(14,7))
    plt.plot(train_data.index, train_data['Close_next'], label='学習期間 実測値', color='blue')
    plt.plot(train_data.index, train_data['Predicted_Close'], label='学習期間 予測値', color='blue', linestyle='--')
    plt.plot(test_data.index, y_test, label='テスト期間 実測値', color='orange')
    plt.plot(test_data.index, test_data['Predicted_Close'], label='テスト期間 予測値', color='orange', linestyle='--')
    
    # 誤差指標テキストを右上に表示
    textstr = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%"
    plt.gca().text(0.98, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.title(f"{ticker} - Actual vs Predicted Close Price")
    plt.xlabel("日付")
    plt.ylabel("終値 (円)")
    plt.legend()
    plt.grid(True)
    
    st.pyplot(plt)
