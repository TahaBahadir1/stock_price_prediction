
# 📈 Stock Price Prediction with PyTorch: LSTM vs GRU



### 🎯 Project Overview
This project focuses on predicting the daily closing stock price of **Apple Inc. (AAPL)** using Deep Learning techniques in **PyTorch**. Two popular Recurrent Neural Network (RNN) architectures—**LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)**—are trained, evaluated, and compared against a **Naive Baseline** model using a 20-day sliding window ve multi-feature (OHLCV) approach.

> ⚠️️ **Disclaimer:** This project is strictly for educational and research purposes and does not constitute financial or investment advice.

### ✨ Key Features
* **Multi-Feature Input:** Utilizes 5 core market features (`Open`, `High`, `Low`, `Close`, `Volume`) rather than relying solely on univariate closing prices.
* **Leakage-Safe Preprocessing:** Chronological train/test split ($80\% / 20\%$) where the `MinMaxScaler` is fitted **exclusively** on the training data to prevent data leakage.
* **Comprehensive EDA:** Includes statistical summaries, Simple Moving Averages ($20$, $50$, and $200$-day SMA), trading volume analysis, and feature correlation heatmaps.
* **Regularization:** Implements $20\%$ Dropout across recurrent layers to mitigate overfitting.
* **Benchmarking:** Evaluates deep learning models against a **Naive Baseline** ($\hat{y}_t = y_{t-1}$) using multiple regression metrics ($MSE$, $RMSE$, $MAE$, $R^2$).

---

### 🛠️ Dataset & Preprocessing
* **Source:** Yahoo Finance (`yfinance`)
* **Ticker:** `AAPL` (Apple Inc.)
* **Date Range:** `2019-01-02` to `2023-12-29` ($1,258$ trading days)
* **Sliding Window:** $20$ days (predicting the $21\text{st}$ day's `Close` price)
* **Training Set:** $1,006$ days ($986$ sequences)
* **Test Set:** $252$ days ($232$ sequences)

---

### 🧠 Model Architectures & Hyperparameters
Both models were trained under identical conditions for a fair comparison:
* **Input Size:** $5$ features
* **Hidden Size:** $64$ units
* **Number of Layers:** $2$ stacked recurrent layers
* **Dropout:** $0.2$
* **Optimizer:** Adam ($\text{lr} = 0.001$)
* **Loss Function:** Mean Squared Error ($MSE$)
* **Batch Size:** $64$ | **Epochs:** $100$

---

### 📊 Results & Comparison

| Model | MSE (USD) | RMSE (USD) | MAE (USD) | $R^2$ Score | Training Time (s) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | \$28.31 | \$5.32 | \$4.57 | 0.8591 | 153.15s | 51,521 |
| **GRU** | \$10.88 | \$3.30 | \$2.74 | 0.9458 | 172.51s | 38,657 |
| **Naive Baseline** | **\$4.61** | **\$2.15** | **\$1.66** | **0.9770** | 0.00s | 0 |

#### 💡 Key Takeaways:
1. **GRU Outperformed LSTM:** With $\sim 25\%$ fewer parameters ($38.6\text{K}$ vs $51.5\text{K}$), the GRU model generalized significantly better on the test set, achieving an $R^2$ of **0.9458** and reducing MAE from **\$4.57** to **\$2.74**.
2. **The Reality of Financial Time Series (Naive Baseline):** The Naive Baseline (predicting tomorrow's price as today's price) achieved the lowest error. In financial markets, stock prices often follow a *random walk*, making single-step price level prediction notoriously difficult without macroeconomic indicators, technical oscillators, or sentiment analysis.

---

### 🚀 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install required dependencies
pip install numpy pandas matplotlib seaborn scikit-learn torch yfinance jupyter

# Launch Jupyter Notebook
jupyter notebook
```


* Modele RSI, MACD, Bollinger Bantları gibi teknik indikatörlerin eklenmesi.
* Attention mekanizması ve Transformer tabanlı zaman serisi modellerinin denenmesi.
* Optuna ile hiperparametre optimizasyonu yapılması.
