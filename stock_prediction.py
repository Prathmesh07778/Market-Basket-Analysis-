import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import datetime
import warnings
import seaborn as sns
from matplotlib.dates import DateFormatter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class StockPricePredictor:
    def __init__(self, ticker="AAPL", years=3):
        """Initialize the stock price predictor with ticker and data range."""
        self.ticker = ticker
        self.years = years
        self.df = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance."""
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365*self.years)
        
        print(f"Fetching {self.years} years of historical data for {self.ticker}...")
        self.df = yf.download(self.ticker, start=start_date, end=end_date)
        
        if self.df.empty or len(self.df) < 100:
            raise ValueError(f"Insufficient data available for {self.ticker}")
            
        print(f"Downloaded {len(self.df)} days of data")
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data and engineer features."""
        print("Preprocessing data and engineering features...")
        
        # Basic preprocessing
        self.df.dropna(inplace=True)
        
        # Feature engineering - Technical indicators
        # Moving averages
        self.df['MA5'] = self.df['Close'].rolling(window=5).mean()
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA50'] = self.df['Close'].rolling(window=50).mean()
        
        # Price momentum
        self.df['Momentum5'] = self.df['Close'] - self.df['Close'].shift(5)
        self.df['Momentum10'] = self.df['Close'] - self.df['Close'].shift(10)
        
        # Volatility (standard deviation)
        self.df['Volatility'] = self.df['Close'].rolling(window=20).std()
        
        # Return rates
        self.df['DailyReturn'] = self.df['Close'].pct_change()
        self.df['5DayReturn'] = self.df['Close'].pct_change(5)
        
        # Exponential moving average
        self.df['EMA12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA26'] = self.df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Relative Strength Index (RSI) - simplified calculation
        delta = self.df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands - Simple implementation using Series objects
        rolling_mean = self.df['Close'].rolling(window=20).mean()
        rolling_std = self.df['Close'].rolling(window=20).std()
        self.df['BB_Middle'] = rolling_mean
        self.df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        self.df['BB_Lower'] = rolling_mean - (rolling_std * 2)
        
        # Lag features (previous days' closing prices)
        for i in range(1, 6):
            self.df[f'Lag_{i}'] = self.df['Close'].shift(i)
        
        # Drop rows with NaN values after feature creation
        self.df.dropna(inplace=True)
        
        return self.df
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis."""
        print("Performing exploratory data analysis...")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        
        # Plot 1: Stock price history with moving averages
        self.df['Close'].plot(ax=axes[0], color='blue', label='Close Price')
        self.df['MA20'].plot(ax=axes[0], color='red', label='20-day MA')
        self.df['MA50'].plot(ax=axes[0], color='green', label='50-day MA')
        axes[0].set_title(f'{self.ticker} Price History with Moving Averages', fontsize=14)
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Volume
        self.df['Volume'].plot(ax=axes[1], color='purple', alpha=0.8)
        axes[1].set_title(f'{self.ticker} Trading Volume', fontsize=14)
        axes[1].set_ylabel('Volume', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Technical indicators
        self.df['RSI'].plot(ax=axes[2], color='orange', label='RSI')
        axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.5)
        axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        axes[2].set_title('Relative Strength Index (RSI)', fontsize=14)
        axes[2].set_ylabel('RSI Value', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Format dates on x-axis
        date_form = DateFormatter("%Y-%m")
        for ax in axes:
            ax.xaxis.set_major_formatter(date_form)
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_EDA.png")
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        selected_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD']
        correlation = self.df[selected_cols].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title(f'{self.ticker} Feature Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_correlation.png")
        plt.close()
        
        # Distribution of daily returns
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['DailyReturn'].dropna(), kde=True, bins=50)
        plt.title(f'{self.ticker} Daily Returns Distribution', fontsize=14)
        plt.xlabel('Daily Return', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_returns_dist.png")
        plt.close()
        
    def prepare_train_test_data(self, test_size=0.2):
        """Prepare training and testing datasets."""
        print("Preparing training and testing datasets...")
        
        # Define feature columns to use
        feature_cols = ['MA5', 'MA20', 'MA50', 'Momentum5', 'Momentum10', 
                      'Volatility', 'DailyReturn', 'MACD', 'RSI', 
                      'Lag_1', 'Lag_2', 'Lag_3']
        
        # Scale features
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        
        scaled_features = self.scaler_features.fit_transform(self.df[feature_cols])
        scaled_target = self.scaler_target.fit_transform(self.df[['Close']])
        
        # Create dataset with scaled values
        dataset = pd.DataFrame(scaled_features, index=self.df.index, columns=feature_cols)
        dataset['Close'] = scaled_target
        
        # Split into train and test
        split_idx = int(len(dataset) * (1 - test_size))
        train_data = dataset.iloc[:split_idx]
        test_data = dataset.iloc[split_idx:]
        
        # Prepare X and y for train and test sets
        X_train = train_data[feature_cols].values
        y_train = train_data['Close'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['Close'].values
        
        return X_train, y_train, X_test, y_test, train_data, test_data
    
    def train_linear_regression(self, X_train, y_train):
        """Train a linear regression model."""
        print("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['linear_regression'] = model
        return model
    
    def train_arima(self):
        """Train an ARIMA model on the closing prices."""
        print("Training ARIMA model...")
        
        # Prepare time series data
        train_size = int(0.8 * len(self.df))
        train_data = self.df['Close'].iloc[:train_size]
        
        try:
            # For quick implementation, use a standard p,d,q configuration
            model = ARIMA(train_data, order=(5,1,0))
            model_fit = model.fit()
            self.models['arima'] = model_fit
            return model_fit
        except Exception as e:
            print(f"ARIMA model training failed: {e}")
            return None
    
    def make_predictions(self, X_test, test_data):
        """Make predictions using trained models."""
        print("Generating predictions...")
        predictions = {}
        
        # Linear Regression predictions
        if 'linear_regression' in self.models:
            lr_pred = self.models['linear_regression'].predict(X_test)
            lr_pred = self.scaler_target.inverse_transform(lr_pred.reshape(-1, 1)).flatten()
            predictions['linear_regression'] = pd.Series(lr_pred, index=test_data.index)
        
        # ARIMA predictions
        if 'arima' in self.models:
            # Forecast for the test period
            forecast_steps = len(test_data)
            arima_forecast = self.models['arima'].forecast(steps=forecast_steps)
            predictions['arima'] = pd.Series(arima_forecast, index=test_data.index)
        
        self.predictions = predictions
        return predictions
    
    def calculate_metrics(self, test_data):
        """Calculate performance metrics for the models."""
        print("Calculating performance metrics...")
        metrics = {}
        actual = self.df.loc[test_data.index, 'Close']
        
        for model_name, predictions in self.predictions.items():
            # Drop any NaN values before calculating metrics
            valid_idx = ~(pd.isna(predictions) | pd.isna(actual))
            clean_actual = actual[valid_idx]
            clean_pred = predictions[valid_idx]
            
            # Skip if there's no valid data
            if len(clean_actual) == 0:
                print(f"Warning: No valid data for {model_name} metrics calculation")
                continue
                
            rmse = np.sqrt(mean_squared_error(clean_actual, clean_pred))
            mae = mean_absolute_error(clean_actual, clean_pred)
            mape = mean_absolute_percentage_error(clean_actual, clean_pred)
            
            metrics[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape * 100  # Convert to percentage
            }
        
        self.metrics = metrics
        return metrics
    
    def forecast_future(self, days=10):
        """Generate future forecasts."""
        print(f"Generating {days}-day forecast...")
        future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')
        forecasts = {}
        
        # Linear Regression forecast
        if 'linear_regression' in self.models:
            # Get the most recent feature values
            latest_features = self.df[['MA5', 'MA20', 'MA50', 'Momentum5', 'Momentum10', 
                                      'Volatility', 'DailyReturn', 'MACD', 'RSI', 
                                      'Lag_1', 'Lag_2', 'Lag_3']].iloc[-1].values.reshape(1, -1)
            
            # Scale features
            latest_features_scaled = self.scaler_features.transform(latest_features)
            
            # Initialize arrays to store future predictions
            future_preds = []
            current_features = latest_features_scaled.copy()
            
            # Recursively predict future values
            close_values = self.df['Close'].values
            for _ in range(days):
                # Predict next closing price
                next_close_scaled = self.models['linear_regression'].predict(current_features)[0]
                next_close = self.scaler_target.inverse_transform([[next_close_scaled]])[0][0]
                future_preds.append(next_close)
                
                # Update features for next prediction (simplified approach)
                # In a real implementation, you would update each feature more accurately
                
                # Update lag features (shift previous predictions)
                current_features[0, -1] = current_features[0, -2]  # Lag_3 = old Lag_2
                current_features[0, -2] = current_features[0, -3]  # Lag_2 = old Lag_1
                current_features[0, -3] = next_close_scaled  # Lag_1 = new scaled prediction
                
                # For demonstration, we're simplifying the feature updates
                # In a real implementation, you would calculate MA, momentum, etc. properly
            
            forecasts['linear_regression'] = pd.Series(future_preds, index=future_dates)
        
        # ARIMA forecast
        if 'arima' in self.models:
            arima_forecast = self.models['arima'].forecast(steps=days)
            forecasts['arima'] = pd.Series(arima_forecast, index=future_dates)
        
        return forecasts
    
    def plot_predictions(self, test_data):
        """Plot the actual vs predicted values."""
        print("Plotting predictions...")
        actual = self.df.loc[test_data.index, 'Close']
        
        plt.figure(figsize=(14, 7))
        plt.plot(actual.index, actual, 'b-', label='Actual Price', linewidth=2)
        
        colors = ['r-', 'g-', 'm-', 'c-']
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            plt.plot(predictions.index, predictions, colors[i % len(colors)], 
                     label=f'{model_name} Predictions', linewidth=1.5, alpha=0.8)
        
        plt.title(f'{self.ticker} Stock Price Prediction', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_predictions.png")
        plt.close()
    
    def plot_future_forecast(self, forecasts):
        """Plot future forecasts."""
        print("Plotting future forecasts...")
        
        # Get the last 30 days of actual data
        recent_data = self.df['Close'].iloc[-30:]
        
        plt.figure(figsize=(14, 7))
        # Plot recent actual data
        plt.plot(recent_data.index, recent_data, 'b-', label='Historical Price', linewidth=2)
        
        # Plot forecast lines for each model with different colors
        colors = ['r-', 'g-', 'm-', 'c-']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            plt.plot(forecast.index, forecast, colors[i % len(colors)], 
                     label=f'{model_name} Forecast', linewidth=2, alpha=0.8)
            
            # Add confidence interval (simplified - just +/- 5%)
            upper = forecast * 1.05
            lower = forecast * 0.95
            plt.fill_between(forecast.index, lower, upper, color=colors[i % len(colors)], alpha=0.2)
        
        # Format x-axis to show dates clearly
        plt.gcf().autofmt_xdate()
        
        plt.title(f'{self.ticker} Stock Price Forecast', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add vertical line separating historical data from forecasts
        plt.axvline(x=self.df.index[-1], color='k', linestyle='--', alpha=0.7)
        plt.annotate('Forecast Start', (self.df.index[-1], self.df['Close'].iloc[-1]),
                    xytext=(10, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_forecast.png")
        plt.close()
    
    def generate_report(self, forecasts):
        """Generate a summary report."""
        print("\n===== STOCK PREDICTION SUMMARY REPORT =====")
        print(f"Ticker: {self.ticker}")
        print(f"Analysis Period: {self.df.index[0].strftime('%Y-%m-%d')} to {self.df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total Days Analyzed: {len(self.df)}")
        
        print("\nPERFORMANCE METRICS:")
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.upper()} MODEL")
            print(f"  RMSE: ${metrics['RMSE']:.2f}")
            print(f"  MAE: ${metrics['MAE']:.2f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
        
        print("\nFUTURE PRICE FORECASTS:")
        for model_name, forecast in forecasts.items():
            print(f"\n{model_name.upper()} FORECAST")
            for date, price in forecast.items():
                print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")
        
        print("\nKEY FINDINGS:")
        # Find best performing model based on RMSE
        best_model = min(self.metrics, key=lambda x: self.metrics[x]['RMSE'])
        print(f"- Best performing model: {best_model.upper()} (RMSE: ${self.metrics[best_model]['RMSE']:.2f})")
        
        # Calculate recent price change
        recent_return = ((self.df['Close'].iloc[-1] / self.df['Close'].iloc[-30]) - 1) * 100
        print(f"- Recent 30-day price change: {recent_return:.2f}%")
        
        # Get last price
        last_price = self.df['Close'].iloc[-1]
        print(f"- Last closing price: ${last_price:.2f}")
        
        # Calculate predicted price change
        next_week_price = forecasts[best_model].iloc[4] if len(forecasts[best_model]) > 4 else forecasts[best_model].iloc[-1]
        price_change = ((next_week_price / last_price) - 1) * 100
        direction = "increase" if price_change > 0 else "decrease"
        print(f"- Projected 5-day {direction}: {abs(price_change):.2f}%")
        
        print("\nTECHNICAL INDICATORS (CURRENT):")
        last_row = self.df.iloc[-1]
        print(f"- RSI: {last_row['RSI']:.2f}")
        if last_row['RSI'] > 70:
            print("  Interpretation: Potentially overbought")
        elif last_row['RSI'] < 30:
            print("  Interpretation: Potentially oversold")
        else:
            print("  Interpretation: Neutral")
        
        print(f"- MACD: {last_row['MACD']:.4f}")
        macd_signal = "bullish" if last_row['MACD'] > last_row['MACD_Signal'] else "bearish"
        print(f"  Signal: {macd_signal}")
        
        print("\n===== END OF REPORT =====")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        self.fetch_data()
        self.preprocess_data()
        self.perform_eda()
        
        X_train, y_train, X_test, y_test, train_data, test_data = self.prepare_train_test_data()
        
        # Train models
        self.train_linear_regression(X_train, y_train)
        self.train_arima()
        
        # Make predictions
        predictions = self.make_predictions(X_test, test_data)
        
        # Calculate metrics
        metrics = self.calculate_metrics(test_data)
        
        # Plot predictions
        self.plot_predictions(test_data)
        
        # Generate future forecasts
        forecasts = self.forecast_future(days=10)
        
        # Plot future forecasts
        self.plot_future_forecast(forecasts)
        
        # Generate report
        self.generate_report(forecasts)
        
        return forecasts


# Run the analysis
if __name__ == "__main__":
    # Allow the user to input the ticker symbol
    ticker = input("Enter stock ticker symbol (default is AAPL): ") or "AAPL"
    
    # Create predictor and run analysis
    predictor = StockPricePredictor(ticker=ticker, years=3)
    try:
        forecasts = predictor.run_full_analysis()
        print(f"\nAnalysis complete! Check the generated graphs ({ticker}_*.png) for visualizations.")
    except Exception as e:
        print(f"Error during analysis: {e}")