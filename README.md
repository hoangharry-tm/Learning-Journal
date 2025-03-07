<a name="top"></a>

# 🧸 My Learning Journal

_Welcome to my __journal__! Here, I'm writing my daily progress on how I studied to success._
_My current concentration (as of March 2, 2025) is learning quantitative finance or algorithmic trading._
_Here is the guide to my journal: [Quantitative Finance/Algorithmic Trading](./algo-trading/algo-trading.md)_

## Roadmap

The overall structure of the roadmap:

| Phases |                                      Details                                      |
| :----: | :-------------------------------------------------------------------------------: |
| __1__  | 📌 Understanding the Basics of Trading & Finance [⤵](#Phase-1-Trading-and-Finance) |
| __2__  |          🪴 Learning & Setting Up Your Stack [⤵](#Phase-2-Learning-Code)           |
| __3__  |     🧪 Research & Strategy Development [⤵](#Phase-3-Research-and-Development)      |
| __4__  |         🚀 Start Coding the Live Trading System [⤵](#Phase-4-Live-Trading)         |
| __5__  |            🧑🏻‍🚀 Going Live & Scaling [⤵](#Phase-5-Going-Live-and-Scaling)            |

<details>

<summary><h3 id="Phase-1-Trading-and-Finance">📌 Phase 1: <i>Understanding the Basics of Trading & Finance</i></h3></summary>

__🎯 Goal:__ Learn the fundamentals of financial markets, trading strategies, and risk management before writing code.  

#### __🔍 What to Learn__

- __Market Structure & Participants__  
  - How exchanges, brokers, and market makers work.  
  - Order types: market orders, limit orders, stop-loss, etc.  
  - Liquidity, bid-ask spread, and slippage.  

- __Trading Strategies & Concepts__  
  - Mean reversion vs. momentum strategies.  
  - Arbitrage (statistical arbitrage, triangular arbitrage).  
  - Market microstructure & HFT strategies.  

- __Risk Management & Portfolio Construction__  
  - Position sizing, stop-loss, and hedging.  
  - Risk-adjusted return metrics (Sharpe, Sortino ratios).  
  - Modern Portfolio Theory (MPT), Kelly Criterion.  

#### __📚 Recommended Resources__

- 📖 _"Quantitative Trading"_ – Ernest Chan  
- 📖 _"Algorithmic Trading"_ – Ernest Chan  
- 📖 _"Market Microstructure Theory"_ – Maureen O’Hara  
- 🖥️ _YouTube: QuantInsti, AlgoTrading101_  

#### __🏆 Milestones__

✅ Understand different market participants and trading mechanics.  
✅ Be able to explain at least __two__ trading strategies in detail.  
✅ Know how to evaluate risk vs. reward in a strategy.  

</details>

<details>

<summary><h3 id="Phase-2-Learning-Code">🪴 Phase 2: <i>Learning & Setting Up Your Stack</i></h3></summary>

__🎯 Goal:__ Learn __Clojure + Python__, set up market data storage, connect to trading APIs, and structure your system.  

#### __🔍 What to Learn__

- __Programming Languages__  
  - __Clojure__: Functional programming, concurrency, data structures.  
  - __Python__: Data science, ML libraries, visualization.  

- __Market Data Handling__  
  - Data ingestion (from APIs, databases).  
  - Storing historical data in __PostgreSQL__ or __Redis__.  

- __Brokerage API Integration__  
  - __Interactive Brokers (IBKR)__: REST API vs. TWS API.  
  - Setting up __real-time data feeds__ & executing orders.  

- __Technology Stack__  
  - __Kafka__ (event streaming).  
  - __Flare__ or __Onyx__ (Clojure-based data processing).  

#### __📚 Recommended Resources__

- 📖 _"Clojure for the Brave and True"_ – Daniel Higginbotham  
- 📖 _"Living Clojure"_ – Carin Meier  
- 🖥️ _IBKR API Documentation_  
- 🖥️ _PostgreSQL, Redis, Kafka Tutorials_  

#### __🏆 Milestones__

✅ Be comfortable with basic Clojure syntax & functional programming.  
✅ Store __market data__ in a database for later use.  
✅ Connect to __IBKR API__ and fetch real-time data.  

</details>

<details>

<summary><h3 id="Phase-3-Research-and-Development">🧪 Phase 3: <i>Research & Strategy Development</i></h3></summary>

__🎯 Goal:__ Develop, test, and validate __trading strategies__ using historical data.  

#### __🔍 What to Learn__

- __Backtesting & Simulation__  
  - Use __backtest.clj__ (Clojure) or __backtrader__ (Python).  
  - Ensure __slippage, transaction costs, and latency__ are simulated.  

- __Risk Management & Portfolio Optimization__  
  - Implement __stop-loss, max drawdown, volatility targeting__.  
  - Optimize strategy parameters using __Bayesian Optimization__.  

- __Machine Learning in Trading__  
  - Feature engineering from financial data.  
  - Use __Scikit-Learn (Python)__ for regression/classification models.  
  - Explore deep learning models for predictive trading.  

#### __📚 Recommended Resources__

- 📖 _"Advances in Financial Machine Learning"_ – Marcos López de Prado  
- 🖥️ _QuantConnect & Backtrader Tutorials_  
- 🖥️ _Machine Learning for Trading (Google Cloud, FastAI)_  

#### __🏆 Milestones__

✅ Run a __backtest__ of at least one trading strategy.  
✅ Implement risk management measures (stop-loss, drawdown control).  
✅ Train a basic __ML model__ for predictive analytics.  

</details>

<details>

<summary><h3 id="Phase-4-Live-Trading">🚀 Phase 4: <i>Start Coding the Live Trading System</i></h3></summary>

__🎯 Goal:__ Implement a __real-time trading system__, optimize latency, and automate order execution.  

#### __🔍 What to Learn__

- __Live Execution Architecture__  
  - Build a __real-time event-driven trading system__.  
  - Implement __order books, real-time price monitoring__.  

- __Latency Optimization__  
  - Use __async & multithreading__ (core.async in Clojure).  
  - Kernel tuning (Linux networking stack optimization).  

- __Production Deployment__  
  - Deploy on a __low-latency cloud provider__ (AWS, DigitalOcean).  
  - Monitor __execution slippage and transaction costs__.  

#### __📚 Recommended Resources__

- 📖 _"Designing Data-Intensive Applications"_ – Martin Kleppmann  
- 🖥️ _Low-Latency Systems (Clojure & JVM tuning guides)_  
- 🖥️ _IBKR Paper Trading API for testing_  

#### __🏆 Milestones__

✅ Deploy a __real-time execution system__ that can place orders.  
✅ Optimize order execution for __low latency & minimal slippage__.  
✅ Automate __risk checks & monitoring__.  

</details>

<details>

<summary><h3 id="Phase-5-Going-Live-and-Scaling">📈 Phase 5: <i>Going Live & Scaling</i></h3></summary>

__🎯 Goal:__ Deploy a __fully operational system__, optimize performance, and scale up trading capital.  

#### __🔍 What to Learn__

- __Performance Monitoring & Logging__  
  - Track PnL, slippage, risk exposure.  
  - Use __Grafana__ for real-time dashboards.  

- __Scaling Strategies__  
  - __Cloud Scaling__ – AWS, DigitalOcean, Kubernetes.  
  - Deploy __multiple strategies across asset classes__.  

- __HFT Optimizations (if applicable)__  
  - FPGA-based order execution (if latency-critical).  
  - Co-located servers near __exchange data centers__.  

#### __📚 Recommended Resources__

- 📖 _"Inside the Black Box"_ – Rishi Narang  
- 🖥️ _Monitoring & Logging (Prometheus, Grafana)_  
- 🖥️ _AWS High-Performance Computing for Finance_  

#### __🏆 Milestones__

✅ Your system __runs live & executes trades__ automatically.  
✅ Performance monitoring detects anomalies & logs all transactions.  
✅ Strategies scale to __higher capital amounts with risk control__.  

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

<!--How the Stock Market Works: A Beginner's Guide to Investment-->

<!--"Clojure for the Brave and True" – Daniel Higginbotham-->
