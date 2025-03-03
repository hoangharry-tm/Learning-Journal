# 🧸 My Learning Journal

_Welcome to my __journal__! Here, I'm writing my daily progress on how I studied to success._
_My current concentration (as of March 2, 2025) is learning quantitative finance or algorithmic trading._
_Here is the guide to my journal: [Quantitative Finance/Algorithmic Trading](./algo-trading/algo-trading.md)_

## Table of content

1. Roadmap [🔗](#roadmap)

## Roadmap

<!-- ChatGPT Roadmap
# **📍 Full Roadmap: Algorithmic Trading with Clojure & Python**
### *(Low Latency, Machine Learning, and Scalable Data Engineering)*  

## **Phase 1: Learn Trading Fundamentals (2-4 weeks)**
📚 **Same as before** – Understand market structures, order types, strategies, and risk management.  

---

## **Phase 2: Learn Clojure & Python for Trading (3-6 weeks)**  
💡 **Why Clojure?**  
- **Immutable, functional programming** → Great for handling financial data streams.  
- **JVM-based** → High performance & integrates well with existing trading infra.  
- **Concurrency & parallelism** → **core.async** for message passing, **Clojure reducers** for parallel computing.  
- **Interoperability** → Connects with Java libraries (e.g., Interactive Brokers API).  

### **✅ Clojure Topics to Learn (For Execution & Data Engineering)**  
✅ Functional programming (Lisp macros, higher-order functions)  
✅ Concurrency & Parallelism (core.async, reducers, GraalVM)  
✅ Streaming & Event-Driven Systems (Kafka, Onyx, core.async)  
✅ Low-latency execution (clj-ib-client, FIX API)  

💻 *Example: Concurrency in Clojure using core.async*  
```clojure
(require '[clojure.core.async :as async])
(def ch (async/chan))
(async/go (println "Received order:" (async/<! ch)))
(async/>!! ch {:symbol "AAPL" :price 150.0})
```

### **✅ Python Topics to Learn (For Machine Learning & Backtesting)**  
💡 **Same as before**: Pandas, NumPy, Scikit-Learn, PyTorch, Backtrader, etc.  

📚 **Resources:**  
- 📖 *Clojure for the Brave and True*  
- 🎥 *Clojure in Action*  

---

## **Phase 3: Build Data Infrastructure (4-8 weeks)**  
💡 **Goal:** Stream real-time market data, store it efficiently.  

### **✅ Step 1: Market Data Collection**
🔹 **Use Interactive Brokers API with Clojure**  
- **Library:** [clj-ib-client](https://github.com/stanshel/clj-ib-client) (wrapper around IB API)  
- Alternative: Alpaca API (for stocks), Binance API (for crypto)  

💻 **Fetch Market Data from IBKR in Clojure**  
```clojure
(require '[ib-client.core :as ib])

(def client (ib/start-client {:host "127.0.0.1" :port 7496}))
(ib/request-market-data client {:symbol "AAPL"})
```

🔹 **Stream Data with Kafka (Clojure + core.async + Onyx)**  
```clojure
(require '[clojure-kafka.client :as kafka])
(def producer (kafka/producer {:bootstrap-servers "localhost:9092"}))
(kafka/send producer "market-data" {:symbol "AAPL" :price 150.0})
```

---

### **✅ Step 2: Storing Market Data Efficiently**
| **Database**                           | **Use Case**                                   |
| -------------------------------------- | ---------------------------------------------- |
| **XTDB (Immutable, Event-Sourced DB)** | Storing tick data, order history               |
| **PostgreSQL**                         | Storing metadata (trades, logs, user settings) |
| **ClickHouse**                         | Fast OLAP queries on historical market data    |
| **Parquet + MinIO/S3**                 | Storing historical data for ML training        |

💻 **Example: Store Market Data in XTDB**  
```clojure
(require '[xtdb.api :as xt])

(def node (xt/start-node {}))
(xt/submit-tx node [[:put {:xt/id :AAPL :price 150.0 :timestamp (System/currentTimeMillis)}]])
```

---

## **Phase 4: Data Processing & Feature Engineering (4-6 weeks)**  
💡 **Goal:** Compute technical indicators, features for ML models.  

🔹 **Use Clojure for Streaming & Python for Feature Engineering**  
✅ **Clojure:** Kafka + Onyx for real-time data processing  
✅ **Python:** Pandas, Scikit-learn for feature extraction  

💻 **Streaming Data with Onyx (Clojure Example)**  
```clojure
(require '[onyx.api :as onyx])
(def job {:workflow [[:ingest :process] [:process :output]]})
```

💻 **Feature Engineering in Python**  
```python
import pandas as pd

df['moving_avg'] = df['close'].rolling(window=10).mean()
df['momentum'] = df['close'] - df['close'].shift(10)
```

---

## **Phase 5: Train Machine Learning Models (6-10 weeks)**  
💡 **Goal:** Predict price movements & optimize execution.  

### **Model Choices**
✅ **Time-Series Forecasting:** LSTMs, Transformers, ARIMA  
✅ **Order Flow Imbalance:** Reinforcement Learning, CNNs  
✅ **Mean Reversion & Statistical Arbitrage:** Kalman Filters  

💻 **Python: Train an LSTM Model**  
```python
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 50)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        return self.fc(self.lstm(x)[0])
```

✅ **Deploy models via FastAPI & call from Clojure.**  

---

## **Phase 6: Backtesting (2-4 weeks)**  
💡 **Goal:** Simulate strategies on historical data.  
🔹 **Use Backtrader (Python) for backtesting**  

💻 **Example Strategy in Backtrader**  
```python
import backtrader as bt

class Strategy(bt.Strategy):
    def next(self):
        if self.data.close[0] > self.data.close[-1]:
            self.buy()
```

✅ **Connect Backtrader to Clojure with a REST API.**  

---

## **Phase 7: Trade Execution & Risk Management (4-6 weeks)**  
💡 **Goal:** Execute trades with **low-latency** & **proper risk controls**.  

🔹 **Use FIX API (for speed) or IBKR TWS API**  
| **API**          | **Latency** | **Use Case**                       |
| ---------------- | ----------- | ---------------------------------- |
| **IBKR Web API** | ~250ms      | Slowest                            |
| **IBKR TWS API** | ~100ms      | Medium latency                     |
| **FIX API**      | **~50ms**   | **Best for low-latency execution** |

💻 **Execute Orders via IBKR API (Clojure Example)**  
```clojure
(ib/place-order client {:symbol "AAPL" :action "BUY" :quantity 100})
```

✅ **Use Clojure’s core.async for concurrent order execution.**  

---

## **Phase 8: Monitoring & Logging (2-4 weeks)**  
💡 **Goal:** Track performance, latency, trade execution.  

🔹 **Use Prometheus & Grafana** for visualization  
💻 **Monitor Order Execution Latency in Clojure**  
```clojure
(require '[clojure-prometheus.core :as prometheus])
(def counter (prometheus/counter :order_execution_latency))
(prometheus/inc counter 10)
```

---

### 🚀 **Final Thoughts**
This roadmap will take **6-12 months**, but by the end, you’ll have a **fully automated, low-latency trading system** using **Clojure + Python ML**.  

🔥 **Would you like a more detailed guide on any specific step?** 😊
-->

The overall structure of the roadmap:

| Phases |                        Details                         |
| :----: | :----------------------------------------------------: |
| __1__  | Trading Fundametals [⤵️](#phase-1-trading-fundamentals) |
| __2__  |                   Learn Programming                    |
| __3__  |               Build Data Infrastructure                |
| __4__  |         Data Processing & Feature Engineering          |
| __5__  |                    Machine Learning                    |
| __6__  |                      Backtesting                       |
| __7__  |           Trade Execution & Risk Management            |

### Phase 1: Trading Fundamentals
