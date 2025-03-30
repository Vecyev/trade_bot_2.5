#!/usr/bin/env python
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import optuna
import joblib
import asyncio

# Import configuration from core/config.py
from .config import config

# Setup institutional-grade logging
logger = logging.getLogger(__name__)
logger.setLevel(config.get("LOG_LEVEL", "INFO"))
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Import additional modules
from sklearn.metrics import accuracy_score
from utils.pnl_tracker import PnLTracker
from utils.fetch_real_options_data import fetch_options_data
from strategy.trade_scorer import TradeScorer  # New: import the trade scorer

class BacktestEngine:
    """
    An institutional-grade backtest engine that:
      - Automatically pulls real options data via fetch_options_data
      - Loads 5 years of historical options data from config['data_path']
      - Filters the data to only include options with expiration between 16 and 21 days (DTE)
      - Trains an XGBoost model (with optional hyperparameter tuning via Optuna)
      - Uses the TradeScorer (which combines ML predictions, conviction, and risk adjustments)
        to decide on trade entries
      - Runs backtests by simulating trade signals with dynamic, volatility-adjusted exit criteria
      - Logs performance metrics and workflow details for evaluation
    """

    def __init__(self):
        self.data_path = config.get("data_path", "data/real_options_data.csv")
        self.model_path = config.get("model_path", "models/xgb_model.pkl")
        self.train_model_flag = config.get("train_model", True)
        self.predict_threshold = config.get("predict_threshold", 0.5)
        self.strategy_params = config.get("strategy_params", {})
        self.optuna_trials = config.get("optuna_trials", 25)
        # Base trading parameters (defaults from config)
        self.trade_holding_period = config.get("trade_holding_period", 2)  # base days holding
        self.stop_loss_pct = config.get("stop_loss_pct", 0.03)
        self.take_profit_pct = config.get("take_profit_pct", 0.05)

        self.model = None

        logger.info("[BacktestEngine] Initialized with configuration:")
        logger.info(f"  data_path={self.data_path}")
        logger.info(f"  model_path={self.model_path}")
        logger.info(f"  train_model={self.train_model_flag}")
        logger.info(f"  predict_threshold={self.predict_threshold}")
        logger.info(f"  optuna_trials={self.optuna_trials}")
        logger.info(f"  trade_holding_period={self.trade_holding_period}")
        logger.info(f"  stop_loss_pct={self.stop_loss_pct}")
        logger.info(f"  take_profit_pct={self.take_profit_pct}")

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[BacktestEngine] Data file not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        # Normalize column names to lowercase.
        df.columns = df.columns.str.lower()
        logger.info(f"[BacktestEngine] Loaded {len(df)} rows from {self.data_path}")

        # Compute DTE if "date" and "expiry" exist.
        if "date" in df.columns and "expiry" in df.columns:
            df["dte"] = df.apply(lambda row: 
                                 (datetime.strptime(str(row["expiry"]), "%Y%m%d") - datetime.strptime(row["date"], "%Y-%m-%d")).days,
                                 axis=1)
            initial_count = len(df)
            df = df[(df["dte"] >= 16) & (df["dte"] <= 21)]
            logger.info(f"[BacktestEngine] Filtered data on DTE: kept {len(df)} of {initial_count} rows (16<=DTE<=21)")
        else:
            logger.warning("[BacktestEngine] 'date' or 'expiry' column missing; cannot filter by DTE.")

        # Filter to only numeric columns and 'label'
        if 'label' in df.columns:
            numeric_cols = df.select_dtypes(include=['number', 'bool', 'category']).columns.tolist()
            if 'label' not in numeric_cols:
                numeric_cols.append('label')
            df = df[numeric_cols]
        return df

    def train_model(self, df: pd.DataFrame):
        logger.info("[BacktestEngine] Starting model training...")
        if 'label' not in df.columns:
            raise ValueError("[BacktestEngine] 'label' column missing in dataset for training.")
        X = df.drop('label', axis=1)
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        model_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        model_params.update(self.strategy_params)
        if self.optuna_trials > 0:
            self.run_optuna_tuning(X_train, y_train)
        self.model = xgb.XGBClassifier(**model_params)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        logger.info(f"[BacktestEngine] Model training complete. Test Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"[BacktestEngine] Created directory: {model_dir}")
        joblib.dump(self.model, self.model_path)
        logger.info(f"[BacktestEngine] Model saved to {self.model_path}")

    def run_optuna_tuning(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.info("[BacktestEngine] Starting Optuna hyperparameter tuning...")
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                "eval_metric": "logloss",
                "use_label_encoder": False
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            return accuracy_score(y_train, preds)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.optuna_trials)
        logger.info(f"[BacktestEngine] Optuna best params: {study.best_params}, Best score: {study.best_value:.3f}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[BacktestEngine] Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"[BacktestEngine] Model loaded from {self.model_path}")

    async def run_backtest(self):
        if self.model is None:
            logger.info("[BacktestEngine] Model not in memory; loading from disk...")
            self.load_model()
        df = self.load_data()
        if 'label' in df.columns:
            X = df.drop('label', axis=1)
            y = df['label']
        else:
            logger.warning("[BacktestEngine] No 'label' column found. Using entire dataset as features only.")
            X = df
            y = None
        probas = self.model.predict_proba(X)[:, 1]
        predictions = (probas >= self.predict_threshold).astype(int)
        if y is not None:
            acc = accuracy_score(y, predictions)
            logger.info(f"[BacktestEngine] Backtest Accuracy={acc:.3f} with threshold={self.predict_threshold}")
        # Compute volatility from daily log returns.
        if "price" in df.columns:
            prices = df["price"].values
            if len(prices) > 1:
                log_returns = np.diff(np.log(prices))
                volatility = np.std(log_returns)
            else:
                volatility = 0.01
            logger.info(f"[BacktestEngine] Computed daily volatility: {volatility:.4f}")
            # Dynamically adjust trading parameters based on volatility.
            dynamic_holding = max(1, int(round(self.trade_holding_period / (1 + volatility))))
            dynamic_stop_loss = self.stop_loss_pct * (1 + volatility)
            dynamic_take_profit = self.take_profit_pct * (1 + volatility)
            logger.info(f"[BacktestEngine] Dynamic holding period: {dynamic_holding} day(s), "
                        f"Stop-loss: {dynamic_stop_loss:.3f}, Take-profit: {dynamic_take_profit:.3f}")
            from utils.pnl_tracker import PnLTracker
            pnl_tracker = PnLTracker()
            # Instantiate TradeScorer to incorporate conviction scores.
            from strategy.trade_scorer import TradeScorer
            trade_scorer = TradeScorer(symbol="NVDA")
            # Dummy trade class for simulation.
            class DummyTrade:
                def __init__(self, symbol, strike, expiry, side, entry_price, quantity=1):
                    self.symbol = symbol
                    self.strike = strike
                    self.expiry = expiry
                    self.side = side
                    self.entry_price = entry_price
                    self.quantity = quantity
            # Simulate trades over the dataset.
            for i in range(len(df) - dynamic_holding):
                if predictions[i] == 1:
                    entry_price = df.iloc[i]["price"]
                    # Create a dummy option object to pass to the TradeScorer.
                    dummy_option = DummyTrade(
                        symbol="NVDA", strike=entry_price,
                        expiry=(datetime.now() + timedelta(days=dynamic_holding)).strftime("%Y%m%d"),
                        side="LONG", entry_price=entry_price, quantity=1
                    )
                    # Get hybrid score from the trade scorer.
                    hybrid_score = trade_scorer.score_and_log_trade(dummy_option, premium=entry_price, side="LONG")
                    if hybrid_score < 0.15:
                        continue  # Skip trade if conviction is too low.
                    exit_price = None
                    for j in range(i+1, i + dynamic_holding + 1):
                        future_price = df.iloc[j]["price"]
                        if future_price >= entry_price * (1 + dynamic_take_profit):
                            exit_price = future_price
                            logger.info(f"Take-profit triggered at row {j}: {future_price}")
                            break
                        elif future_price <= entry_price * (1 - dynamic_stop_loss):
                            exit_price = future_price
                            logger.info(f"Stop-loss triggered at row {j}: {future_price}")
                            break
                    if exit_price is None:
                        exit_price = df.iloc[i + dynamic_holding]["price"]
                        logger.info(f"No exit condition met; exiting at row {i + dynamic_holding}: {exit_price}")
                    expiry = (datetime.now() + timedelta(days=dynamic_holding)).strftime("%Y%m%d")
                    trade = DummyTrade(
                        symbol="SIM", strike=entry_price, expiry=expiry,
                        side="LONG", entry_price=entry_price, quantity=1
                    )
                    pnl_tracker.record_trade(trade, entry_price, side="LONG", quantity=1)
                    pnl_tracker.close_trade(trade, exit_price)
            pnl_tracker.report()
        else:
            logger.warning("[BacktestEngine] 'price' column not found in dataset; skipping trade simulation.")
        logger.info("[BacktestEngine] Backtest run complete. (Simulated trade signals and PnL computed.)")

    def run(self):
        logger.info("[BacktestEngine] Updating dataset from real market data...")
        updated_csv = fetch_options_data(ticker_symbol="NVDA", expiration_date=None, output_csv=self.data_path)
        if updated_csv:
            self.data_path = updated_csv
            logger.info(f"[BacktestEngine] Data path updated to {self.data_path}")
        else:
            logger.warning("[BacktestEngine] Failed to update dataset; using existing data.")
        if self.train_model_flag:
            df = self.load_data()
            self.train_model(df)
        else:
            logger.info("[BacktestEngine] Skipping training (train_model=False).")
        asyncio.run(self.run_backtest())
        logger.info("[BacktestEngine] Workflow complete.")

# Standalone usage example.
if __name__ == "__main__":
    def main():
        engine = BacktestEngine()
        engine.run()
    main()
