import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

warnings.filterwarnings('ignore')

# Manual Technical Indicator Functions
def calculate_sma(data, window=14):
    """Calculate Simple Moving Average (SMA)"""
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window=14):
    """Calculate Exponential Moving Average (EMA)"""
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(data):
    """Add SMA, EMA, RSI to dataset"""
    data['SMA'] = calculate_sma(data)
    data['EMA'] = calculate_ema(data)
    data['RSI'] = calculate_rsi(data)
    return data

# Stock Predictor Class with Advanced Features
class StockPredictorLSTM:
    def __init__(self, symbol='TSLA', start_date='2020-01-01', end_date=None):
        """Initialize the predictor with stock symbol and date range"""
        self.symbol = symbol
        self.start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        self.end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d') if end_date is None else pd.to_datetime(end_date).strftime('%Y-%m-%d')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        print(f"Initialized predictor for {symbol}")
        print(f"Date range: {self.start_date} to {self.end_date}")

    def fetch_data(self):
      """Fetch stock data"""
      print(f"\nFetching data for {self.symbol}...")
      self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
      if self.data.empty:
          raise ValueError("No data fetched. Check symbol or date range.")
    
      # Check for NaN values in the dataset
      if self.data.isna().sum().sum() > 0:
          print("Data contains NaN values. Filling missing values...")
          self.data.fillna(method='ffill', inplace=True)  # Forward fill missing values
          # Alternatively, you can drop rows with NaN values:
          # self.data.dropna(inplace=True)
          
      print(f"Successfully fetched {len(self.data)} rows of data.")
      return self.data


    def prepare_data(self, sequence_length=60):
      """Prepare data for LSTM training"""
      print("\nPreparing data...")
      if self.data is None or self.data.empty:
          raise ValueError("No data available. Please fetch data first.")

      # Scale closing prices
      data = self.data[['Open', 'High', 'Low', 'Close']].values  # Use all columns
      scaled_data = self.scaler.fit_transform(data)
      
      # Check if scaling resulted in any NaN values
      if np.isnan(scaled_data).sum() > 0:
          print("Scaling produced NaN values. Please check your data.")
          return None, None, None, None
      
      # Create sequences
      X, y = [], []
      for i in range(sequence_length, len(scaled_data)):
          X.append(scaled_data[i-sequence_length:i])
          y.append(scaled_data[i, 3])  # Predicting the 'Close' price
      
      X, y = np.array(X), np.array(y)
      
      # Split into train and test sets
      train_size = int(len(X) * 0.8)
      X_train, X_test = X[:train_size], X[train_size:]
      y_train, y_test = y[:train_size], y[train_size:]
      
      print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
      return X_train, X_test, y_train, y_test


    def build_model(self, input_shape, layer_type='LSTM', units=50):
        """Build LSTM or GRU model"""
        model = Sequential()
        
        if layer_type == 'LSTM':
            model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        elif layer_type == 'GRU':
            model.add(GRU(units, return_sequences=True, input_shape=input_shape))
        
        model.add(Dropout(0.2))
        
        if layer_type == 'LSTM':
            model.add(LSTM(units, return_sequences=False))
        elif layer_type == 'GRU':
            model.add(GRU(units, return_sequences=False))
        
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
      """Train the LSTM model"""
      print("\nTraining model...")
      
      # Check if NaN values exist in the data
      if np.isnan(X_train).sum() > 0 or np.isnan(y_train).sum() > 0 or np.isnan(X_test).sum() > 0 or np.isnan(y_test).sum() > 0:
          print("Training data contains NaN values. Aborting training.")
          return None
      
      self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
      history = self.model.fit(
          X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=epochs,
          batch_size=batch_size,
          verbose=1
      )
      return history


    def evaluate_model(self, X_test, y_test):
      """Evaluate model performance"""
      print("\nEvaluating model...")
      predictions = self.model.predict(X_test)
      
      # Check for NaN values in predictions
      if np.isnan(predictions).sum() > 0:
          print("Model predictions contain NaN values. Aborting evaluation.")
          return
      
      # Inverse transform predictions and actual values
      predictions = self.scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], X_test.shape[2]-1)), predictions)))
      y_test = self.scaler.inverse_transform(np.hstack((np.zeros((y_test.shape[0], X_test.shape[2]-1)), y_test.reshape(-1, 1))))
      
      # Calculate metrics
      mse = mean_squared_error(y_test, predictions)
      rmse = np.sqrt(mse)
      mae = mean_absolute_error(y_test, predictions)
      r2 = r2_score(y_test, predictions)
      
      print(f"Mean Squared Error (MSE): {mse}")
      print(f"Root Mean Squared Error (RMSE): {rmse}")
      print(f"Mean Absolute Error (MAE): {mae}")
      print(f"R-squared (R2): {r2}")
      
      # Plot results
      plt.figure(figsize=(12, 6))
      plt.plot(y_test, label='Actual Prices', color='blue')
      plt.plot(predictions, label='Predicted Prices', color='red')
      plt.title('Stock Price Prediction')
      plt.xlabel('Time')
      plt.ylabel('Price')
      plt.legend()
      plt.show()


# Main Execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = StockPredictorLSTM(symbol='TSLA', start_date='2022-01-01', end_date='2023-12-01')
    
    # Fetch data
    data = predictor.fetch_data()
    
    # Prepare data (add technical indicators)
    X_train, X_test, y_train, y_test = predictor.prepare_data(sequence_length=60)
    
    # Train model
    history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Evaluate model
    predictor.evaluate_model(X_test, y_test)

    # Load and use saved model (if needed)
    # predictor.load_trained_model('stock_predictor_lstm_final.h5')