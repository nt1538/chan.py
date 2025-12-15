import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import os

# Suppress warnings
warnings.filterwarnings('ignore')

def run_daily_rolling_window_backtest(
    data_path,
    output_dir="./output/daily_rolling_backtest",
    training_window_days=30,
    rolling_period_days=14,
    start_date=None,
    position_size=1.0,
    initial_capital=10000,
    model_params=None
):
    """
    Run a daily rolling window backtest.
    
    Args:
        data_path: Path to the dataset CSV file
        output_dir: Directory to save output files
        training_window_days: Number of days to use for training (default: 30 days)
        rolling_period_days: Number of days to roll forward (default: 14 days)
        start_date: Starting date for backtest (format: "YYYY-MM-DD"), if None will use earliest possible
        position_size: Portion of capital to use per trade (default: 100%)
        initial_capital: Initial capital for backtest (default: $10,000)
        model_params: Dictionary of XGBoost model parameters (will use defaults if None)
    
    Returns:
        DataFrame of daily returns and performance metrics
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Default model parameters
    if model_params is None:
        model_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
    
    print("=" * 80)
    print("Daily Rolling Window Backtesting")
    print("=" * 80)
    print(f"Training window: {training_window_days} days")
    print(f"Rolling period: {rolling_period_days} days")
    print(f"Start date: {start_date if start_date else 'Earliest possible'}")
    print(f"Position size: {position_size * 100}%")
    print(f"Initial capital: ${initial_capital}")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Full dataset date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Identify unique dates in the dataset
    df['date'] = df['timestamp'].dt.date
    unique_dates = df['date'].unique()
    unique_dates = np.sort(unique_dates)  # Sort dates chronologically
    print(f"Number of unique trading days: {len(unique_dates)}")
    
    # Set start date index
    if start_date:
        start_date = pd.to_datetime(start_date).date()
        if start_date in unique_dates:
            start_idx = np.where(unique_dates == start_date)[0][0]
        else:
            # Find the closest date after the specified start date
            valid_dates = unique_dates[unique_dates >= start_date]
            if len(valid_dates) > 0:
                start_idx = np.where(unique_dates == valid_dates[0])[0][0]
            else:
                raise ValueError(f"No valid dates found after specified start date: {start_date}")
    else:
        # Find a good starting point with enough data for the initial window
        start_idx = 0
    
    # Ensure we have enough data for our analysis
    required_days = training_window_days + 2 + rolling_period_days  # training + threshold + test + rolling days
    if start_idx + required_days > len(unique_dates):
        raise ValueError(f"Not enough trading days in dataset. Need at least {required_days} days after start date.")
    
    # Define the dates for the entire backtest period
    backtest_dates = unique_dates[start_idx:start_idx + training_window_days + 2 + rolling_period_days]
    print(f"Backtest period: {backtest_dates[0]} to {backtest_dates[-1]}")
    
    # Identify feature columns
    print("\n[2/5] Identifying feature columns...")
    
    # Exclude metadata and target columns
    exclude_patterns = ['timestamp', 'bsp_type', 'direction', 'profit_target', 
                       'has_profit', 'return_', 'label_', 'target_return_', 
                       'snapshot_', 'klu_idx', 'exit_', 'date']
    
    # Filter to feature columns only
    feature_cols = [col for col in df.columns 
                    if not any(pat in col for pat in exclude_patterns)]
    
    # Remove encoded columns if original exists
    if 'direction' in feature_cols and 'direction_encoded' in feature_cols:
        feature_cols.remove('direction')
    if 'bsp_type' in feature_cols and 'bsp_type_encoded' in feature_cols:
        feature_cols.remove('bsp_type')
    
    feature_cols = sorted(feature_cols)
    print(f"Features identified: {len(feature_cols)}")
    
    # Helper function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
        
        # Calculate accuracy (correct direction prediction)
        correct_direction = np.mean((y_true > 0) == (y_pred > 0)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Accuracy': correct_direction
        }
    
    # Function to optimize thresholds using grid search
    def optimize_thresholds(signals_df):
        """Find optimal buy/sell thresholds for signal data"""
        if len(signals_df) == 0:
            return 0.0, 0.0  # Default thresholds if no signals
        
        # Define threshold ranges
        thresholds = np.linspace(-0.5, 2.0, 26)  # -0.5% to 2.0% in 0.25% steps
        
        best_return = -float('inf')
        best_buy_threshold = 0.0
        best_sell_threshold = 0.0
        
        # Simple grid search
        for buy_threshold in thresholds:
            for sell_threshold in thresholds:
                # Simulate trading with these thresholds
                buy_signals = signals_df[signals_df['direction'] == 'buy']
                sell_signals = signals_df[signals_df['direction'] == 'sell']
                
                # Filter signals based on thresholds
                valid_buys = buy_signals[buy_signals['predicted_profit_pct'] >= buy_threshold]
                valid_sells = sell_signals[sell_signals['predicted_profit_pct'] >= sell_threshold]
                
                # Calculate a score (e.g., average profit of valid signals)
                buy_score = valid_buys['profit_target_pct'].mean() if len(valid_buys) > 0 else 0
                sell_score = valid_sells['profit_target_pct'].mean() if len(valid_sells) > 0 else 0
                
                # Combined score weighted by signal counts
                combined_score = (len(valid_buys) * buy_score + len(valid_sells) * sell_score) / (len(valid_buys) + len(valid_sells)) if (len(valid_buys) + len(valid_sells)) > 0 else 0
                
                # Update best thresholds if better
                if combined_score > best_return:
                    best_return = combined_score
                    best_buy_threshold = buy_threshold
                    best_sell_threshold = sell_threshold
        
        return best_buy_threshold, best_sell_threshold
    
    # Function to backtest a single day with given thresholds
    def backtest_day(signals_df, buy_threshold, sell_threshold, initial_capital=10000, position_size=1.0):
        """Run backtest for a single day with specified thresholds"""
        if len(signals_df) == 0:
            return {
                'final_value': initial_capital,
                'return_pct': 0.0,
                'trades': 0,
                'win_rate': 0.0,
                'signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
            }
        
        # Sort signals chronologically
        signals_df = signals_df.sort_values('timestamp').reset_index(drop=True)
        
        # Strategy state
        capital = initial_capital
        cash = initial_capital
        shares = 0
        entry_price = 0
        trades = []
        
        # Process signals
        for _, row in signals_df.iterrows():
            current_price = row['klu_close']
            predicted_profit = row['predicted_profit_pct']
            direction = row['direction']
            
            # Buy logic
            if direction == 'buy' and shares == 0:
                if predicted_profit >= buy_threshold:
                    trade_capital = cash * position_size
                    shares = trade_capital / current_price
                    entry_price = current_price
                    cash -= trade_capital
                    
                    trades.append({
                        'entry_price': current_price,
                        'direction': 'buy'
                    })
            
            # Sell logic
            elif direction == 'sell' and shares > 0:
                if predicted_profit >= sell_threshold:
                    exit_value = shares * current_price
                    trade_profit = exit_value - (shares * entry_price)
                    trade_return_pct = (trade_profit / (shares * entry_price)) * 100
                    
                    cash += exit_value
                    
                    # Record trade result
                    if trades:
                        trades[-1].update({
                            'exit_price': current_price,
                            'trade_profit': trade_profit,
                            'trade_return_pct': trade_return_pct
                        })
                    
                    # Reset position
                    shares = 0
                    entry_price = 0
        
        # Close any open position at the end of the day
        if shares > 0:
            final_price = signals_df.iloc[-1]['klu_close']
            exit_value = shares * final_price
            trade_profit = exit_value - (shares * entry_price)
            trade_return_pct = (trade_profit / (shares * entry_price)) * 100
            
            cash += exit_value
            
            if trades:
                trades[-1].update({
                    'exit_price': final_price,
                    'trade_profit': trade_profit,
                    'trade_return_pct': trade_return_pct,
                    'note': 'Closed at end of day'
                })
        
        # Calculate metrics
        final_value = cash
        return_pct = ((final_value / initial_capital) - 1) * 100
        
        completed_trades = sum(1 for trade in trades if 'exit_price' in trade)
        winning_trades = sum(1 for trade in trades if 'trade_return_pct' in trade and trade['trade_return_pct'] > 0)
        win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
        
        buy_signals = len(signals_df[signals_df['direction'] == 'buy'])
        sell_signals = len(signals_df[signals_df['direction'] == 'sell'])
        
        return {
            'final_value': final_value,
            'return_pct': return_pct,
            'trades': completed_trades,
            'win_rate': win_rate,
            'signals': len(signals_df),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
        }
    
    # Calculate Buy & Hold return for a specific day
    def calculate_buy_hold(day_df):
        """Calculate Buy & Hold return for a specific day"""
        if len(day_df) == 0:
            return 0.0
        
        # Sort chronologically
        day_df = day_df.sort_values('timestamp').reset_index(drop=True)
        
        # Get first and last prices
        first_price = day_df.iloc[0]['klu_close']
        last_price = day_df.iloc[-1]['klu_close']
        
        # Calculate return
        return ((last_price / first_price) - 1) * 100
    
    # Prepare data structures to store results
    daily_results = []
    model_metrics = []
    
    # Start rolling window backtest
    print("\n[3/5] Starting rolling window backtest...")
    
    # Loop through each day in the rolling period
    for day in tqdm(range(rolling_period_days), desc="Processing days"):
        # Define window indices
        train_start_idx = day
        train_end_idx = day + training_window_days - 1
        threshold_day_idx = train_end_idx + 1
        test_day_idx = threshold_day_idx + 1
        
        # Skip if we don't have enough days left
        if test_day_idx >= len(backtest_dates):
            print(f"  Warning: Not enough days left for complete window at day {day}, skipping...")
            continue
        
        # Extract dates for each period
        train_dates = backtest_dates[train_start_idx:train_end_idx + 1]
        threshold_date = backtest_dates[threshold_day_idx]
        test_date = backtest_dates[test_day_idx]
        
        print(f"\nDay {day+1}/{rolling_period_days}: Testing on {test_date}")
        print(f"  Training period: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
        print(f"  Threshold optimization: {threshold_date}")
        print(f"  Testing: {test_date}")
        
        # Filter data for each period
        train_df = df[df['date'].isin(train_dates)].copy()
        threshold_df = df[df['date'] == threshold_date].copy()
        test_df = df[df['date'] == test_date].copy()
        
        # Check if we have sufficient data in each period
        if len(train_df) < 100:  # Arbitrary minimum sample size
            print(f"  Warning: Insufficient training data ({len(train_df)} samples), skipping day {day}...")
            continue
            
        if len(threshold_df) == 0:
            print(f"  Warning: No data for threshold day {threshold_date}, skipping day {day}...")
            continue
            
        if len(test_df) == 0:
            print(f"  Warning: No data for test day {test_date}, skipping day {day}...")
            continue
        
        # Split data by buy/sell signals
        train_valid = train_df[train_df['has_profit_target'] == 1].copy()
        train_buy = train_valid[train_valid['direction'] == 'buy'].copy()
        train_sell = train_valid[train_valid['direction'] == 'sell'].copy()
        
        threshold_valid = threshold_df[threshold_df['has_profit_target'] == 1].copy()
        threshold_buy = threshold_valid[threshold_valid['direction'] == 'buy'].copy()
        threshold_sell = threshold_valid[threshold_valid['direction'] == 'sell'].copy()
        
        test_valid = test_df[test_df['has_profit_target'] == 1].copy()
        test_buy = test_valid[test_valid['direction'] == 'buy'].copy()
        test_sell = test_valid[test_valid['direction'] == 'sell'].copy()
        
        # Log data sizes
        print(f"  Training data: {len(train_valid)} signals ({len(train_buy)} buy, {len(train_sell)} sell)")
        print(f"  Threshold data: {len(threshold_valid)} signals ({len(threshold_buy)} buy, {len(threshold_sell)} sell)")
        print(f"  Test data: {len(test_valid)} signals ({len(test_buy)} buy, {len(test_sell)} sell)")
        
        # Skip if not enough training data for either buy or sell
        if len(train_buy) < 10 or len(train_sell) < 10:
            print(f"  Warning: Not enough buy/sell training samples, skipping day {day}...")
            continue
        
        # Prepare training data
        X_train_buy = train_buy[feature_cols].fillna(0)
        y_train_buy = train_buy['profit_target_pct']
        
        X_train_sell = train_sell[feature_cols].fillna(0)
        y_train_sell = train_sell['profit_target_pct']
        
        # Train Buy model
        print(f"  Training Buy model...")
        buy_model = xgb.XGBRegressor(**model_params)
        buy_model.fit(X_train_buy, y_train_buy)
        
        # Train Sell model
        print(f"  Training Sell model...")
        sell_model = xgb.XGBRegressor(**model_params)
        sell_model.fit(X_train_sell, y_train_sell)
        
        # Make predictions on threshold day
        if len(threshold_buy) > 0:
            X_threshold_buy = threshold_buy[feature_cols].fillna(0)
            threshold_buy['predicted_profit_pct'] = buy_model.predict(X_threshold_buy)
        
        if len(threshold_sell) > 0:
            X_threshold_sell = threshold_sell[feature_cols].fillna(0)
            threshold_sell['predicted_profit_pct'] = sell_model.predict(X_threshold_sell)
        
        # Combine threshold predictions
        threshold_signals = pd.concat([threshold_buy, threshold_sell], ignore_index=True)
        
        # Optimize thresholds on threshold day
        print(f"  Optimizing thresholds...")
        optimal_buy_threshold, optimal_sell_threshold = optimize_thresholds(threshold_signals)
        print(f"  Optimal thresholds: Buy={optimal_buy_threshold:.2f}%, Sell={optimal_sell_threshold:.2f}%")
        
        # Make predictions on test day
        if len(test_buy) > 0:
            X_test_buy = test_buy[feature_cols].fillna(0)
            test_buy['predicted_profit_pct'] = buy_model.predict(X_test_buy)
        
        if len(test_sell) > 0:
            X_test_sell = test_sell[feature_cols].fillna(0)
            test_sell['predicted_profit_pct'] = sell_model.predict(X_test_sell)
        
        # Combine test predictions
        test_signals = pd.concat([test_buy, test_sell], ignore_index=True)
        
        # Calculate test metrics
        if len(test_buy) > 0:
            buy_metrics = calculate_metrics(test_buy['profit_target_pct'], test_buy['predicted_profit_pct'])
        else:
            buy_metrics = None
            
        if len(test_sell) > 0:
            sell_metrics = calculate_metrics(test_sell['profit_target_pct'], test_sell['predicted_profit_pct'])
        else:
            sell_metrics = None
        
        # Run backtest on test day with optimal thresholds
        print(f"  Running backtest on test day...")
        backtest_result = backtest_day(test_signals, optimal_buy_threshold, optimal_sell_threshold, initial_capital, position_size)
        
        # Calculate Buy & Hold for comparison
        bh_return = calculate_buy_hold(test_df)
        
        # Record results
        daily_result = {
            'day': day + 1,
            'date': test_date,
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'threshold_date': threshold_date,
            'buy_threshold': optimal_buy_threshold,
            'sell_threshold': optimal_sell_threshold,
            'strategy_return': backtest_result['return_pct'],
            'buy_hold_return': bh_return,
            'vs_buy_hold': backtest_result['return_pct'] - bh_return,
            'signals': backtest_result['signals'],
            'trades': backtest_result['trades'],
            'win_rate': backtest_result['win_rate'],
            'final_value': backtest_result['final_value'],
            'buy_signals': backtest_result['buy_signals'],
            'sell_signals': backtest_result['sell_signals'],
        }
        
        daily_results.append(daily_result)
        
        # Record model metrics
        model_metric = {
            'day': day + 1,
            'date': test_date,
            'buy_model_samples': len(train_buy),
            'sell_model_samples': len(train_sell),
        }
        
        # Add buy metrics
        if buy_metrics:
            model_metric.update({
                'buy_rmse': buy_metrics['RMSE'],
                'buy_mae': buy_metrics['MAE'],
                'buy_r2': buy_metrics['R2'],
                'buy_accuracy': buy_metrics['Accuracy'],
            })
            
        # Add sell metrics
        if sell_metrics:
            model_metric.update({
                'sell_rmse': sell_metrics['RMSE'],
                'sell_mae': sell_metrics['MAE'],
                'sell_r2': sell_metrics['R2'],
                'sell_accuracy': sell_metrics['Accuracy'],
            })
            
        model_metrics.append(model_metric)
        
        # Report summary for this day
        print(f"  Test day results:")
        print(f"    Strategy return: {backtest_result['return_pct']:.2f}%")
        print(f"    Buy & Hold return: {bh_return:.2f}%")
        print(f"    Outperformance: {backtest_result['return_pct'] - bh_return:+.2f}%")
        print(f"    Trades: {backtest_result['trades']}, Win rate: {backtest_result['win_rate']:.1f}%")
    
    # Convert results to DataFrames
    daily_results_df = pd.DataFrame(daily_results)
    model_metrics_df = pd.DataFrame(model_metrics)
    
    # Analysis and Visualization
    print("\n[4/5] Analyzing results...")
    
    if len(daily_results_df) > 0:
        # Calculate cumulative returns
        daily_results_df['cumulative_strategy'] = (1 + daily_results_df['strategy_return'] / 100).cumprod() * 100 - 100
        daily_results_df['cumulative_buy_hold'] = (1 + daily_results_df['buy_hold_return'] / 100).cumprod() * 100 - 100
        
        # Overall performance metrics
        final_strategy_return = daily_results_df['cumulative_strategy'].iloc[-1]
        final_buy_hold_return = daily_results_df['cumulative_buy_hold'].iloc[-1]
        
        win_days = sum(daily_results_df['strategy_return'] > 0)
        total_days = len(daily_results_df)
        win_rate = (win_days / total_days * 100) if total_days > 0 else 0
        
        outperformance_days = sum(daily_results_df['strategy_return'] > daily_results_df['buy_hold_return'])
        outperformance_rate = (outperformance_days / total_days * 100) if total_days > 0 else 0
        
        print("\n" + "=" * 80)
        print("DAILY ROLLING WINDOW BACKTEST RESULTS")
        print("=" * 80)
        print(f"Total days analyzed: {total_days}")
        print(f"Strategy cumulative return: {final_strategy_return:.2f}%")
        print(f"Buy & Hold cumulative return: {final_buy_hold_return:.2f}%")
        print(f"Strategy vs. Buy & Hold: {final_strategy_return - final_buy_hold_return:+.2f}%")
        print(f"Win rate (positive return days): {win_rate:.1f}%")
        print(f"Outperformance rate (days beating Buy & Hold): {outperformance_rate:.1f}%")
        
        # Create visualizations
        print("\n[5/5] Creating visualizations...")
        
        # 1. Cumulative returns comparison
        plt.figure(figsize=(12, 6))
        plt.plot(daily_results_df['day'], daily_results_df['cumulative_strategy'], label='Strategy', linewidth=2)
        plt.plot(daily_results_df['day'], daily_results_df['cumulative_buy_hold'], label='Buy & Hold', linewidth=2)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Cumulative Returns: Strategy vs. Buy & Hold', fontsize=14)
        plt.xlabel('Day')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cumulative_returns.png", dpi=150)
        print(f"Saved: {output_dir}/cumulative_returns.png")
        
        # 2. Daily returns comparison
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(daily_results_df)), daily_results_df['strategy_return'], alpha=0.7, label='Strategy')
        plt.bar(range(len(daily_results_df)), daily_results_df['buy_hold_return'], alpha=0.5, label='Buy & Hold')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Daily Returns: Strategy vs. Buy & Hold', fontsize=14)
        plt.xlabel('Day')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/daily_returns.png", dpi=150)
        print(f"Saved: {output_dir}/daily_returns.png")
        
        # 3. Strategy outperformance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(daily_results_df)), 
                daily_results_df['strategy_return'] - daily_results_df['buy_hold_return'],
                color=['green' if x > 0 else 'red' for x in 
                       daily_results_df['strategy_return'] - daily_results_df['buy_hold_return']])
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Daily Strategy Outperformance vs. Buy & Hold', fontsize=14)
        plt.xlabel('Day')
        plt.ylabel('Outperformance (%)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/strategy_outperformance.png", dpi=150)
        print(f"Saved: {output_dir}/strategy_outperformance.png")
        
        # 4. Model metrics over time
        if len(model_metrics_df) > 0 and 'buy_accuracy' in model_metrics_df.columns and 'sell_accuracy' in model_metrics_df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(model_metrics_df['day'], model_metrics_df['buy_accuracy'], label='Buy Model Accuracy', marker='o')
            plt.plot(model_metrics_df['day'], model_metrics_df['sell_accuracy'], label='Sell Model Accuracy', marker='s')
            plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random Guess')
            plt.title('Model Direction Accuracy Over Time', fontsize=14)
            plt.xlabel('Day')
            plt.ylabel('Direction Accuracy (%)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_accuracy.png", dpi=150)
            print(f"Saved: {output_dir}/model_accuracy.png")
        
        # 5. Threshold evolution
        plt.figure(figsize=(12, 6))
        plt.plot(daily_results_df['day'], daily_results_df['buy_threshold'], label='Buy Threshold', marker='o')
        plt.plot(daily_results_df['day'], daily_results_df['sell_threshold'], label='Sell Threshold', marker='s')
        plt.title('Optimal Threshold Evolution', fontsize=14)
        plt.xlabel('Day')
        plt.ylabel('Profit Threshold (%)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/threshold_evolution.png", dpi=150)
        print(f"Saved: {output_dir}/threshold_evolution.png")
        
        # Save results to CSV
        daily_results_df.to_csv(f"{output_dir}/daily_results.csv", index=False)
        model_metrics_df.to_csv(f"{output_dir}/model_metrics.csv", index=False)
        print(f"Saved results to {output_dir}/daily_results.csv and {output_dir}/model_metrics.csv")
        
        # Generate summary report
        with open(f"{output_dir}/backtest_summary.txt", 'w') as f:
            f.write("Daily Rolling Window Backtest Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Training window: {training_window_days} days\n")
            f.write(f"  Rolling period: {rolling_period_days} days\n")
            f.write(f"  Position size: {position_size * 100}%\n")
            f.write(f"  Initial capital: ${initial_capital}\n\n")
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"  Total days analyzed: {total_days}\n")
            f.write(f"  Strategy cumulative return: {final_strategy_return:.2f}%\n")
            f.write(f"  Buy & Hold cumulative return: {final_buy_hold_return:.2f}%\n")
            f.write(f"  Strategy vs. Buy & Hold: {final_strategy_return - final_buy_hold_return:+.2f}%\n")
            f.write(f"  Win rate (positive return days): {win_rate:.1f}%\n")
            f.write(f"  Outperformance rate (days beating Buy & Hold): {outperformance_rate:.1f}%\n\n")
            
            f.write("DAILY RESULTS:\n")
            for i, row in daily_results_df.iterrows():
                f.write(f"  Day {row['day']} ({row['date']}): Strategy {row['strategy_return']:.2f}%, ")
                f.write(f"B&H {row['buy_hold_return']:.2f}%, Diff {row['vs_buy_hold']:+.2f}%, ")
                f.write(f"Thresholds: Buy={row['buy_threshold']:.2f}%, Sell={row['sell_threshold']:.2f}%\n")
            
            f.write("\nANALYSIS:\n")
            strategy_better = "OUTPERFORMED" if final_strategy_return > final_buy_hold_return else "UNDERPERFORMED"
            f.write(f"The daily rolling window strategy {strategy_better} Buy & Hold over the {rolling_period_days}-day period ")
            f.write(f"by {final_strategy_return - final_buy_hold_return:+.2f}%.\n")
            
        print(f"Saved summary to {output_dir}/backtest_summary.txt")
        
        # Copy output to user-accessible location
        os.makedirs("/mnt/user-data/outputs/daily_rolling_backtest", exist_ok=True)
        os.system(f"cp -r {output_dir}/* /mnt/user-data/outputs/daily_rolling_backtest/")
        
        print("\n" + "=" * 80)
        print("Daily Rolling Window Backtest Complete!")
        print("=" * 80)
        
        print(f"\n📈 All results saved to: /mnt/user-data/outputs/daily_rolling_backtest/")
        
        return daily_results_df
    else:
        print("No valid daily results data to analyze")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "./output/enhanced_bs_features/SPY_testing_dataset_21.1.1-12.31.csv"  # Path to your dataset
    OUTPUT_DIR = "./output/daily_rolling_backtest"
    
    # Parameters
    TRAINING_WINDOW_DAYS = 30  # Use 1 month of data for training
    ROLLING_PERIOD_DAYS = 200   # Number of days to roll forward
    POSITION_SIZE = 1.0        # Use 100% of capital per trade
    INITIAL_CAPITAL = 10000    # Initial capital
    
    # Run backtest
    results_df = run_daily_rolling_window_backtest(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        training_window_days=TRAINING_WINDOW_DAYS,
        rolling_period_days=ROLLING_PERIOD_DAYS,
        position_size=POSITION_SIZE,
        initial_capital=INITIAL_CAPITAL
    )