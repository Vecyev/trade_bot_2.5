#!/usr/bin/env python
"""
fetch_real_options_data.py

A deployment-ready script to fetch real options chain data from Yahoo Finance.
It processes the data to create a CSV file with key option trade features that support
ML model training and backtesting in our institutional-grade trade bot.
Note: The synthetic fields (delta, yield, etc.) are placeholders and should be refined
for production use with proper financial models.
"""

import os
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime

# Setup robust logging
logger = logging.getLogger("RealOptionsDataCollector")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def fetch_options_data(ticker_symbol="NVDA", expiration_date=None, output_csv="data/real_options_data.csv"):
    """
    Fetches options chain data for the specified ticker from Yahoo Finance.
    
    Args:
        ticker_symbol (str): Stock ticker symbol.
        expiration_date (str): Expiration date in "YYYY-MM-DD" format.
            If None, uses the first available expiration date.
        output_csv (str): Path where the CSV file will be saved.
    
    Returns:
        str: Path to the saved CSV file, or None if an error occurred.
    """
    try:
        logger.info(f"Fetching options data for {ticker_symbol}...")
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        if not expirations:
            logger.error(f"No options expiration dates found for {ticker_symbol}.")
            return None
        
        if expiration_date is None:
            expiration_date = expirations[0]  # Use first available expiration
            logger.info(f"No expiration date provided. Using: {expiration_date}")
        elif expiration_date not in expirations:
            logger.error(f"Provided expiration date {expiration_date} not available. Options available: {expirations}")
            return None
        
        option_chain = ticker.option_chain(expiration_date)
        calls_df = option_chain.calls
        puts_df = option_chain.puts
        
        # Add OptionType column
        calls_df["OptionType"] = "CALL"
        puts_df["OptionType"] = "PUT"
        
        # Combine calls and puts into one DataFrame
        options_df = pd.concat([calls_df, puts_df], ignore_index=True)
        
        # Rename 'lastPrice' to 'price' for consistency with our trade bot
        options_df.rename(columns={"lastPrice": "price"}, inplace=True)
        
        # Add a 'Date' column with today's date for backtesting reference
        options_df["Date"] = datetime.now().strftime("%Y-%m-%d")
        
        # ---- Synthetic Feature Engineering (placeholders) ----
        num_rows = options_df.shape[0]
        # In production, replace these with real calculations:
        options_df["delta"] = 0.3  # Placeholder: in practice, calculate from option greeks
        options_df["yield_to_strike"] = options_df["price"] / options_df["strike"] * 0.05  # Synthetic yield
        options_df["Premium"] = options_df["price"]  # Using price as premium for simplicity
        options_df["ROC"] = options_df["Premium"] / options_df["strike"]
        options_df["RSI"] = 50  # Neutral RSI placeholder
        options_df["Momentum"] = 0.0  # Placeholder momentum
        options_df["IV_percentile"] = options_df["impliedVolatility"].fillna(0.5)  # Use impliedVolatility as proxy
        options_df["NearEarnings"] = 0  # Placeholder; replace with real earnings check if available
        
        # For labeling, a simple heuristic: label = 1 if price is within 95% of strike (indicating an attractive premium)
        options_df["label"] = (options_df["price"] > options_df["strike"] * 0.95).astype(int)
        # ---------------------------------------------------------
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        options_df.to_csv(output_csv, index=False)
        logger.info(f"Real options data saved to {output_csv}")
        return output_csv
    except Exception as e:
        logger.error(f"Error fetching options data for {ticker_symbol}: {e}")
        return None

def main():
    csv_path = fetch_options_data(ticker_symbol="NVDA", expiration_date=None, output_csv="data/real_options_data.csv")
    if csv_path:
        print(f"Data saved to {csv_path}")
    else:
        print("Data fetching failed.")

if __name__ == "__main__":
    main()
