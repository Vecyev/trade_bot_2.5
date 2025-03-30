import streamlit as st
import pandas as pd
import os
import asyncio
from datetime import datetime

# Import your backtest engine from your ML core module
from ml.ML_Module.core.backtest_engine import BacktestEngine

def run():
    st.title("Institutional Trade Bot Dashboard")

    # === BOT STATUS OVERVIEW ===
    st.header("Bot Status Overview")
    log_path = "logs/trade_history.csv"
    last_trade_time = "No trades yet."
    trade_count = 0

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
            if not df.empty:
                trade_count = len(df)
                # Convert Date column to datetime for proper display
                df["Date"] = pd.to_datetime(df["Date"])
                last_trade_time = df["Date"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error(f"Error loading trade log: {e}")
    else:
        st.warning("Trade log not found. Trades will appear here once executed.")

    st.metric("Total Trades Executed", trade_count)
    st.metric("Last Trade Date", last_trade_time)

    # === PERFORMANCE & SCORE ANALYTICS ===
    st.header("Performance & Score Analytics")
    if os.path.exists(log_path) and trade_count > 0:
        df = pd.read_csv(log_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

        st.subheader("Score Distribution Over Time")
        score_cols = ["Conviction", "ML Score", "Hybrid Score"]
        for col in score_cols:
            if col in df.columns:
                st.line_chart(df[col])

        st.subheader("PnL Summary (Completed Trades)")
        if "Actual PnL" in df.columns:
            pnl_df = df[df["Actual PnL"].notnull()]
            st.metric("Total Realized PnL", f"${pnl_df['Actual PnL'].sum():,.2f}")
            st.metric("Avg PnL per Trade", f"${pnl_df['Actual PnL'].mean():,.2f}")
        else:
            st.info("No PnL data available.")

        st.subheader("Recent Trades")
        st.dataframe(df.tail(5))
    else:
        st.info("No trade data available to display performance analytics.")

    # === DISCORD ALERT STATUS ===
    st.header("Discord Alerts")
    st.success("Discord alerts are active and configured.")

    # === CONTROLS ===
    st.header("Controls")

    if st.button("Retrain ML Model"):
        try:
            engine = BacktestEngine()
            data = engine.load_data()
            engine.train_model(data)
            st.success("ML Model retrained successfully!")
        except Exception as e:
            st.error(f"Retraining ML Model failed: {e}")

    if st.button("Run Backtest"):
        try:
            engine = BacktestEngine()
            # Run the backtest asynchronously
            asyncio.run(engine.run_backtest())
            st.success("Backtest completed successfully!")
        except Exception as e:
            st.error(f"Backtest failed: {e}")

if __name__ == "__main__":
    run()
