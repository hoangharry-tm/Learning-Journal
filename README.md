<a name="top"></a>

# 🧸 My Learning Journal

_Welcome to my __journal__! Here, I'm writing my daily progress on how I studied to success._
_My current concentration (as of March 2, 2025) is learning quantitative finance or algorithmic trading._
_Here is the guide to my journal: [Quantitative Finance/Algorithmic Trading](./algo-trading/algo-trading.md)_

## Table of content

- [Roadmap](#roadmap)
  - [Phase 1: _Trading Fundamentals_](#phase-1-trading-fundamentals)
  - [Phase 2: _Learn Programming_](#phase-2-learn-programming)
- [Resources](#resources)

## Roadmap

<!-- ChatGPT Roadmap
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
| __1__  | Trading Fundametals [⤵](#phase-1-trading-fundamentals) |
| __2__  |   Learn Programming [⤵](#phase-2-learn-programming)    |
| __3__  |               Build Data Infrastructure                |
| __4__  |         Data Processing & Feature Engineering          |
| __5__  |                    Machine Learning                    |
| __6__  |                      Backtesting                       |
| __7__  |           Trade Execution & Risk Management            |

<details>
<summary><h3><ins>Phase 1: <i>Trading Fundamentals</i></ins></h3></summary>

#### Study the market structure

In this particular stage, I will learn the foundation of the financial market.
This includes learning about stocks, forex, crypto, bonds, options, etc. Here is
the book to learn from
[How the Stock Market Works : A Beginner's Guide to Investment][Book 1]

#### Learn Market Data & Order Types

_Q_: What is market data?

_A_: Market data consists of

- Level 1 Data: Price, Volume, Bid-Ask Spread
- Level 2 Data (Order Book): Shows buy/sell orders @ different price levels
- And many more!

__Resources__: Learn how to put orders from a specific brokerage (_Interactive Brokers_,
_Quest Trade_, _Binance_, etc.)

#### Learn Trading Strategies

In this particular step, I will try to learn different quantitative speculation
strategies as well as how to leverage information from the movement of the market
to predict patterns.

__Resources__: 📚 _"Algorithmic Trading"_ by Ernest Chen

#### Understand Risk Management

Why is risk management important?

- Avoids blowing up your account from large losses.
- Keeps drawdowns small and manageable.
- Helps maximize long-term profitability.
  
1️⃣ Position Sizing

- Use 1-2% risk per trade to limit losses.
- Adjust position size based on account balance.

2️⃣ Stop-Loss & Take-Profit

- Place stop-losses based on technical indicators (e.g., ATR).
- Take profits at predetermined risk-reward ratios.

3️⃣ Leverage & Margin

- Leverage = Borrowing money to increase trade size.
- Too much leverage = High risk of liquidation.
- Safe leverage: 1:2 for stocks, 1:10 for forex, 1:5 for crypto.

#### Learn How To Read Financial Data

These are some notable data that needed to be learned:

- Candle Stick Charts
- Volume Analysis
- Order Book Analysis

__Resources__: 📖 _"Technical Analysis of the Financial Markets"_ by John Murphy

#### Paper Trading

Applying the knowledge learned to real-world scenario!
I expect to have a trading account set up, some analysis of the prefered market
as well as some well-constructed portfolio to be executed.

#### 🪴 _Action Plan of Phase 1_

- Week 1: Learn market structure, order types, and market data.
- Week 2: Study common algo trading strategies & risk management.
- Week 3: Learn technical & order book analysis.
- Week 4: Open a paper trading account & place simulated trades.

🚀 [_Back to top_](#top)

</details>

<details>
<summary><h3><ins>Phase 2: <i>Learn Programming</i></ins></h3></summary>

Since in this project, I specifically want to use _Clojure_ and _Python_, this
phase is the opportunity for me to learn both.

<details>

<summary>🪴 Week 1: <i>Clojure & Python for Algorithmic Trading</i></summary>

<br/>

→ Learn the basics of Clojure:

- Learn Clojure syntax. REPL, immutability, and functional programming.
- Write simple scripts to process numbers, strings, and collections.

Resources?

- 📚 "Clojure for the Brave and True" – Daniel Higginbotham [(_Online_)][Clojure Book 1]

Some hands-on 🫨:

- [ ] Install Clojure CLI & Leiningen.
- [ ] Run simple __map, filter, and reduce__ functions

→ Then, try to learn Python basics for Data Science & Machine Learning!

- Learn NumPy, Pandas, and Matplotlib for market data analysis.
- Write basic data processing scripts for stock price visualization.

Some resouces for to check out:

- 📖 "Python for Data Analysis" – Wes McKinney

And, again, some hands-on:

- [ ] Load _historical market_ data using `pandas` and visualize price trends.

→ Ultimately, let's learn more about Clojure & Python interoperability.

- Use libpython-clj to run Python code from Clojure.
- Call machine learning models from Python in Clojure pipelines.

Why not try...

- [ ] Writing a Clojure script that call a Python function for a simple calculation?

</details>

<details>

<summary>🌱 Week 2: <i>Market data handling & storage</i></summary>

<br/>

→ Market data sources & API's

- Learn about Interactive Brokers (IBKR) API, Alpha Vantage, Binance, Yahoo Finance.
- Fetch real-time & historical price data.

A book suggestion,

- 📖 "Mastering Python for Finance" – James Ma Weiming

Checkpoint:

- [ ] Write a script to __pull stock data__ from the IBKR API

→ Databases for Algo Trading

- Use PostgreSQL for storing price data and trade logs.
- Use Redis for real-time caching.

📚 Resources:

- 🎥 📺 PostgreSQL for Trading (Youtube)
- 📖 "Designing Data-Intensive Applications" – Martin Kleppmann

Hands-on:

- [ ] Set up a PostgreSQL database and store historical price data.

→ Data Pipelines & ETL

- Build a Clojure-based data pipeline to clean and process market data.
- Use Apache Kafka for streaming live market data.

📚 Resources:

- 🎥 📺 Kafka for Real-Time Trading
- 📖 "Kafka: The Definitive Guide" – Neha Narkhede

Hands-on:

- [ ] Stream live Binance data into PostgreSQL using Kafka.

</details>

</details>

## Resources

### Books

Consider the list of following ebooks providers which cooperate with Queen's
University:

- ProQuest
- Cengage
- Wiley Online Library
- O'Reilly Media

<!--Variables-->

<!--How the Stock Market Works : A Beginner's Guide to Investment-->
[Book 1]: https://ocul-qu.primo.exlibrisgroup.com/view/action/uresolver.do?operation=resolveService&package_service_id=16351423430005158&institutionId=5158&customerId=5150&VE=true

<!--"Clojure for the Brave and True" – Daniel Higginbotham-->
[Clojure Book 1]: https://www.braveclojure.com/
