import pandas as pd
import numpy as np

def calculate_metrics(data, lookback_period, lookahead_period):
    """
    Calculates historical and future high and low metrics for a given DataFrame with OHLC crypto data.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data with date as the index.
        lookback_period (int): Look-back period for calculating historical high and low metrics.
        lookahead_period (int): Look-forward period for calculating future high and low metrics.

    Returns:
        pd.DataFrame: Original DataFrame with additional columns for calculated metrics.
    """

    if {'high', 'low'} not in data.columns:
        raise ValueError("DataFrame must contain 'high' and 'low' columns.")

    data[f'high_{lookback_period}d_max'] = data['high'].rolling(window=lookback_period).max()
    data[f'low_{lookback_period}d_min'] = data['low'].rolling(window=lookback_period).min()

    data[f'high_{lookahead_period}d_future_max'] = data['high'].shift(-lookahead_period).rolling(window=lookahead_period).max()
    data[f'low_{lookahead_period}d_future_min'] = data['low'].shift(-lookahead_period).rolling(window=lookahead_period).min()

    data.dropna(inplace=True)

    return data


# Calculate Highest, Lowest, and Percentage Difference Metrics


class CryptoOHLCMetrics:
    """
    A class for calculating historical and future metrics on day-wise OHLC crypto data.
    Assumes data is sorted in descending order with most recent date first.
    All days and percentage differences are returned as positive values.
    """

    def __init__(self, data: pd.DataFrame, look_back: int, look_forward: int):
        """
        Initialize with OHLC data and set the look-back and look-forward periods.

        Parameters:
        - data (pd.DataFrame): DataFrame containing OHLC data with date as index and 'high', 'low', 'close' columns
        - look_back (int): Number of days for historical metrics calculations (variable1)
        - look_forward (int): Number of days for future metrics calculations (variable2)
        """
        if not {'high', 'low', 'close'}.issubset(data.columns):
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")

        # Ensure data is sorted in descending order
        self.data = data.sort_index(ascending=False)
        self.look_back = look_back
        self.look_forward = look_forward

    def calculate_metrics(self) -> pd.DataFrame:
        """Add all required metrics to the DataFrame."""
        self._historical_high_low()
        self._historical_high_low_differences()
        self._future_high_low()
        self._future_high_low_differences()
        return self.data

    def _historical_high_low(self):
        """Calculate historical high and low metrics over the look-back period."""
        # For descending data, we need to reverse the rolling window
        reversed_data = self.data.iloc[::-1]

        high_series = reversed_data['high'].rolling(window=self.look_back, min_periods=1).max()
        low_series = reversed_data['low'].rolling(window=self.look_back, min_periods=1).min()

        # Reverse back to original order
        self.data[f'High_Last_{self.look_back}_Days'] = high_series.iloc[::-1]
        self.data[f'Low_Last_{self.look_back}_Days'] = low_series.iloc[::-1]

    def _historical_high_low_differences(self):
        """Calculate days since high/low and percentage differences within look-back limit."""
        reversed_data = self.data.iloc[::-1]

        # Calculate days since high
        def days_since_high(window):
            if len(window) == 0:
                return None
            max_idx = window.idxmax()
            days = abs((window.index[0] - max_idx).days)  # Using abs() here
            return min(days, self.look_back)

        # Calculate days since low
        def days_since_low(window):
            if len(window) == 0:
                return None
            min_idx = window.idxmin()
            days = abs((window.index[0] - min_idx).days)  # Using abs() here
            return min(days, self.look_back)

        # Apply rolling calculations and reverse back
        days_since_high_series = reversed_data['high'].rolling(window=self.look_back, min_periods=1).apply(
            days_since_high, raw=False
        )
        days_since_low_series = reversed_data['low'].rolling(window=self.look_back, min_periods=1).apply(
            days_since_low, raw=False
        )

        self.data[f'Days_Since_High_Last_{self.look_back}_Days'] = days_since_high_series.iloc[::-1]
        self.data[f'Days_Since_Low_Last_{self.look_back}_Days'] = days_since_low_series.iloc[::-1]

        # Calculate percentage differences using absolute values
        self.data[f'%_Diff_From_High_Last_{self.look_back}_Days'] = (
            ((self.data['close'] - self.data[f'High_Last_{self.look_back}_Days']) /
            self.data[f'High_Last_{self.look_back}_Days']) * 100
        )

        self.data[f'%_Diff_From_Low_Last_{self.look_back}_Days'] = (
            ((self.data['close'] - self.data[f'Low_Last_{self.look_back}_Days']) /
            self.data[f'Low_Last_{self.look_back}_Days']) * 100
        )

    def _future_high_low(self):
        """Calculate future high and low metrics over the look-forward period."""
        reversed_data = self.data.iloc[::-1]

        # Calculate future highs and lows using forward-looking windows
        future_high = reversed_data['high'].shift(-1).rolling(window=self.look_forward, min_periods=1).max()
        future_low = reversed_data['low'].shift(-1).rolling(window=self.look_forward, min_periods=1).min()

        self.data[f'High_Next_{self.look_forward}_Days'] = future_high.iloc[::-1]
        self.data[f'Low_Next_{self.look_forward}_Days'] = future_low.iloc[::-1]

    def _future_high_low_differences(self):
        """Calculate percentage differences from future high and low within look-forward limit."""
        # Calculate percentage differences using absolute values
        self.data[f'%_Diff_From_High_Next_{self.look_forward}_Days'] = (
            ((self.data['close'] - self.data[f'High_Next_{self.look_forward}_Days']) /
            self.data[f'High_Next_{self.look_forward}_Days']) * 100
        )

        self.data[f'%_Diff_From_Low_Next_{self.look_forward}_Days'] = (
            ((self.data['close'] - self.data[f'Low_Next_{self.look_forward}_Days']) /
            self.data[f'Low_Next_{self.look_forward}_Days']) * 100
        )

# Example usage
# if __name__ == "__main__":
#     # Create sample data
#     dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
#     data = {
#         'high': [100 + i + (i % 3) * 2 for i in range(30)],
#         'low': [95 + i - (i % 3) for i in range(30)],
#         'close': [97 + i + (i % 2) for i in range(30)]
#     }
#     df = pd.DataFrame(data, index=dates).sort_index(ascending=False)

#     # Calculate metrics
#     calculator = CryptoOHLCMetrics(df, look_back=7, look_forward=5)
#     result = calculator.calculate_metrics()
#     print(result)