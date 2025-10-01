import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BSPReturnPredictor:
    def __init__(self, target_horizon='return_5'):
        """
        BSP Return Prediction System
        
        Args:
            target_horizon: Which return to predict ('return_1', 'return_5', 'return_10', 'return_20')
        """
        self.target_horizon = target_horizon
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.results = {}
        
        # Initialize models
        self.init_models()
        
    def init_models(self):
        """Initialize various ML models"""
        
        # Ensemble Methods (Usually best for tabular data)
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        self.models['catboost'] = cb.CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Linear Models (Good for interpretability)
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['lasso'] = Lasso(alpha=0.1)
        self.models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        # Neural Network
        self.models['mlp'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42
        )
        
        # SVM
        self.models['svr'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
    def prepare_features(self, df):
        """Prepare features for ML training"""
        
        # Select relevant features
        feature_cols = [
            # Technical indicators
            'macd_value', 'macd_dif', 'macd_dea', 'rsi', 'kdj_k', 'kdj_d', 'kdj_j',
            'dmi_plus', 'dmi_minus', 'dmi_adx', 'atr', 'atr_ratio',
            
            # Price features
            'price_change_pct', 'high_low_spread_pct', 'upper_shadow', 'lower_shadow', 'body_size',
            
            # BSP specific features
            'feat_divergence_rate', 'feat_bsp1_bi_amp', 'feat_bsp1_bi_klu_cnt', 'feat_bsp1_bi_amp_rate',
            'feat_macd_value', 'feat_macd_dea', 'feat_macd_diff', 'feat_ppo', 'feat_rsi',
            
            # Moving averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'ema_50',
            'price_above_sma_5', 'price_above_sma_10', 'price_above_sma_20', 'price_above_sma_50',
            'price_above_ema_12', 'price_above_ema_26', 'price_above_ema_50',
            
            # Volume features
            'volume_volume_spike', 'volume_accumulation', 'volume_distribution',
            
            # Pattern features
            'candle_doji', 'candle_hammer', 'candle_shooting_star', 'price_breakout_up', 'price_breakout_down',
            
            # BSP type and direction
            'is_buy', 'bsp_type'
        ]
        
        # Handle BSP type encoding
        if 'bsp_type' in df.columns:
            df['bsp_type_encoded'] = pd.Categorical(df['bsp_type']).codes
            feature_cols.append('bsp_type_encoded')
        
        # Add lag features for key indicators
        for col in ['rsi', 'macd_value', 'feat_divergence_rate']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag2'] = df[col].shift(2)
                feature_cols.extend([f'{col}_lag1', f'{col}_lag2'])
        
        # Add rolling statistics
        for col in ['rsi', 'macd_value']:
            if col in df.columns:
                df[f'{col}_roll_mean_5'] = df[col].rolling(5).mean()
                df[f'{col}_roll_std_5'] = df[col].rolling(5).std()
                feature_cols.extend([f'{col}_roll_mean_5', f'{col}_roll_std_5'])
        
        # Filter available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        return df[feature_cols].fillna(0)
    
    def time_series_split_validation(self, df, n_splits=5):
        """Perform time series cross-validation"""
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Prepare features and target
        X = self.prepare_features(df_sorted)
        y = df_sorted[self.target_horizon].fillna(0)
        
        # Remove rows where target is missing
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(X)//10)
        
        results = {}
        for model_name in self.models.keys():
            results[model_name] = {
                'mse_scores': [],
                'mae_scores': [],
                'r2_scores': [],
                'predictions': [],
                'actual': [],
                'test_dates': []
            }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"Processing fold {fold + 1}/{n_splits}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features for models that need it
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            for model_name, model in self.models.items():
                try:
                    # Use scaled features for certain models
                    if model_name in ['mlp', 'svr', 'ridge', 'lasso', 'elastic_net']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[model_name]['mse_scores'].append(mse)
                    results[model_name]['mae_scores'].append(mae)
                    results[model_name]['r2_scores'].append(r2)
                    results[model_name]['predictions'].extend(y_pred)
                    results[model_name]['actual'].extend(y_test)
                    
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    continue
        
        self.results = results
        return results
    
    def train_final_models(self, df):
        """Train final models on full dataset"""
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Use first 80% for training, rest for final validation
        split_idx = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        X_train = self.prepare_features(train_df)
        y_train = train_df[self.target_horizon].fillna(0)
        X_test = self.prepare_features(test_df)
        y_test = test_df[self.target_horizon].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        final_results = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"Training final {model_name} model...")
                
                if model_name in ['mlp', 'svr', 'ridge', 'lasso', 'elastic_net']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate final metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                final_results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test.values
                }
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        self.scalers['final'] = scaler
        return final_results
    
    def print_results_summary(self, cv_results=None, final_results=None):
        """Print summary of model performance"""
        
        if cv_results:
            print("\n" + "="*60)
            print("CROSS-VALIDATION RESULTS")
            print("="*60)
            
            summary_df = []
            for model_name, results in cv_results.items():
                if results['mse_scores']:
                    summary_df.append({
                        'Model': model_name,
                        'Avg MSE': np.mean(results['mse_scores']),
                        'Avg MAE': np.mean(results['mae_scores']),
                        'Avg R²': np.mean(results['r2_scores']),
                        'MSE Std': np.std(results['mse_scores']),
                        'MAE Std': np.std(results['mae_scores']),
                        'R² Std': np.std(results['r2_scores'])
                    })
            
            summary_df = pd.DataFrame(summary_df).sort_values('Avg R²', ascending=False)
            print(summary_df.round(4))
        
        if final_results:
            print("\n" + "="*60)
            print("FINAL MODEL RESULTS (80/20 split)")
            print("="*60)
            
            final_df = []
            for model_name, results in final_results.items():
                final_df.append({
                    'Model': model_name,
                    'MSE': results['mse'],
                    'MAE': results['mae'],
                    'R²': results['r2']
                })
            
            final_df = pd.DataFrame(final_df).sort_values('R²', ascending=False)
            print(final_df.round(4))
    
    def plot_predictions(self, model_name, final_results):
        """Plot predictions vs actual values"""
        
        if model_name not in final_results:
            print(f"Model {model_name} not found in results")
            return
        
        results = final_results[model_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(results['actual'], results['predictions'], alpha=0.6)
        ax1.plot([results['actual'].min(), results['actual'].max()], 
                [results['actual'].min(), results['actual'].max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Returns')
        ax1.set_ylabel('Predicted Returns')
        ax1.set_title(f'{model_name} - Predictions vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # Time series plot
        ax2.plot(results['actual'], label='Actual', alpha=0.7)
        ax2.plot(results['predictions'], label='Predicted', alpha=0.7)
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Returns')
        ax2.set_title(f'{model_name} - Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model_name, top_n=20):
        """Plot feature importance for tree-based models"""
        
        if model_name not in self.feature_importance:
            print(f"Feature importance not available for {model_name}")
            return
        
        importance_df = self.feature_importance[model_name].head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def predict_new_data(self, new_data, model_name='xgboost'):
        """Make predictions on new data"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        X_new = self.prepare_features(new_data)
        
        if model_name in ['mlp', 'svr', 'ridge', 'lasso', 'elastic_net']:
            X_new_scaled = self.scalers['final'].transform(X_new)
            predictions = self.models[model_name].predict(X_new_scaled)
        else:
            predictions = self.models[model_name].predict(X_new)
        
        return predictions

# Usage example
def run_bsp_prediction_pipeline(df_path, target_horizon='return_5'):
    """
    Complete pipeline for BSP return prediction
    
    Args:
        df_path: Path to your CSV file
        target_horizon: Which return to predict
    """
    
    # Load data
    df = pd.read_csv(df_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} BSP records")
    print(f"Predicting: {target_horizon}")
    
    # Initialize predictor
    predictor = BSPReturnPredictor(target_horizon=target_horizon)
    
    # Run cross-validation
    print("\nRunning time series cross-validation...")
    cv_results = predictor.time_series_split_validation(df, n_splits=5)
    
    # Train final models
    print("\nTraining final models...")
    final_results = predictor.train_final_models(df)
    
    # Print results
    predictor.print_results_summary(cv_results, final_results)
    
    # Find best model
    best_model = max(final_results.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest performing model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
    
    # Plot results for best model
    predictor.plot_predictions(best_model[0], final_results)
    
    # Plot feature importance
    if best_model[0] in predictor.feature_importance:
        predictor.plot_feature_importance(best_model[0])
    
    return predictor, cv_results, final_results

# Example usage:
# predictor, cv_results, final_results = run_bsp_prediction_pipeline('your_bsp_data.csv', 'return_5')