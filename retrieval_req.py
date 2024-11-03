import requests
import pandas as pd
from datetime import datetime, timedelta

# CryptoCompare API_Key
API_KEY = '57819025e42cfd2d2d1a716f2e3feb8b0b251ee88ee386cc71f74efaba89e9ed'

def fetch_crypto_ohlc_range(crypto_symbol, currency, timeframe='day', start_date=None, end_date=None):
    """
    Fetch historical OHLC data for a cryptocurrency within a specified date range.

    Parameters:
    - crypto_symbol (str): Symbol of the cryptocurrency (e.g., 'BTC')
    - currency (str): The currency to get OHLC data in (e.g., 'USD', 'INR')
    - timeframe (str): The timeframe for the data. Options are 'day', 'hour', 'minute'.
    - start_date (str): The start date in 'YYYY-MM-DD' format.
    - end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame containing historical OHLC data within the specified range.
    """

    # Define the URL based on the timeframe
    if timeframe == 'day':
        url = 'https://min-api.cryptocompare.com/data/v2/histoday'
    elif timeframe == 'hour':
        url = 'https://min-api.cryptocompare.com/data/v2/histohour'
    elif timeframe == 'minute':
        url = 'https://min-api.cryptocompare.com/data/v2/histominute'
    else:
        raise ValueError("Invalid timeframe. Use 'day', 'hour', or 'minute'.")

    # Convert start and end dates to timestamps
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp()) if start_date else None
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()) if end_date else int(datetime.now().timestamp())

    all_data = []  

    
    while True:
        # Set the parameters for the request
        params = {
            'fsym': crypto_symbol,      
            'tsym': currency,           
            'limit': 2000,              
            'api_key': API_KEY,         
            'toTs': end_timestamp       
        }

        # Send the request
        response = requests.get(url, params=params)
        data = response.json()

        # Check if response contains data
        if data['Response'] != 'Success':
            raise ValueError(f"Error fetching data: {data.get('Message', 'No data available')}")

        # Convert the data to a DataFrame
        ohlc_data = data['Data']['Data']
        df = pd.DataFrame(ohlc_data)

        # Convert the timestamp to a datetime format
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Filter out data outside of the start date
        if start_timestamp:
            df = df[df['time'].apply(lambda x: x.timestamp()) >= start_timestamp]

        # Break if no more data or all data is within range
        if df.empty or (start_timestamp and df['time'].iloc[0].timestamp() <= start_timestamp):
            all_data.append(df)
            break

        # Append data and update the end timestamp for the next batch
        all_data.append(df)
        end_timestamp = int(df['time'].iloc[0].timestamp()) - 1  # Continue from before the earliest timestamp

    # Concatenate all data into a single DataFrame
    final_df = pd.concat(all_data).reset_index(drop=True)

    # Select only relevant columns and rename
    final_df = final_df[['time', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto']]
    final_df.rename(columns={'volumefrom': 'volume', 'volumeto': 'volume_in_currency'}, inplace=True)

    return final_df

# # Example usage
# crypto_symbol = 'BTC'       # Cryptocurrency symbol (e.g., 'BTC')
# currency = 'USD'            # Currency symbol (e.g., 'USD' or 'INR')
# timeframe = 'day'           # Timeframe ('day', 'hour', or 'minute')
# start_date = '2023-01-01'   # Start date in 'YYYY-MM-DD' format
# end_date = '2023-12-31'     # End date in 'YYYY-MM-DD' format

# # Fetch the data
# df = fetch_crypto_ohlc_range(crypto_symbol, currency, timeframe, start_date, end_date)
# print(df.head())


# 2. Using Kraken API_Key 

def get_ohlc_data(api_url, pair, interval, since):
  """
  Fetches OHLC data from the Kraken API for a specific currency pair.

  Args:
      api_url (str): Base URL of the Kraken API endpoint.
      pair (str): Currency pair (e.g., 'LTCUSD', 'BTCUSD').
      interval (int): Time interval in minutes (e.g., 1440 for daily).
      since (int): Unix timestamp for the starting point of data retrieval.

  Returns:
      pd.DataFrame: DataFrame containing OHLC data with columns:
          date, open, high, low, close, volume.

  Raises:
      Exception: If an error occurs during the API request.
  """

  payload = {
      'pair': pair,
      'interval': interval,
      'since': since
  }

  headers = {
      'Accept': 'application/json'
  }

  try:
    response = requests.get(api_url, headers=headers, params=payload)
    data = response.json()

    if data['error'] == []:  # No errors
      ohlc_data = data['result'][f"{pair}Z{pair}"]

      # Convert data to DataFrame and set column names
      df = pd.DataFrame(ohlc_data, columns=[
          'date', 'open', 'high', 'low', 'close', 'volume', 'Count', 'VWAP'
      ])

      # Convert 'Time' column from timestamp to readable date
      df['date'] = pd.to_datetime(df['date'], unit='s')
      df.set_index('date', inplace=True)
      df.drop(['Count', 'VWAP'], axis=1, inplace=True)

      return df

    else:
      raise Exception("Error in API request:", data['error'])

  except Exception as e:
    print(f"An error occurred: {e}")
    return None


# Example usage
# api_url = "https://api.kraken.com/0/public/OHLC"
# pair = 'LTCUSD'
# interval = 1440  # Daily data
# since = 1451606400  # Example timestamp

# df = get_ohlc_data(api_url, pair, interval, since)

# if df is not None:
#   print(df)

