import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


"""
Lable Encoding - 
new parameter added to make the model understand each crypto exchange 

"""
def encode_coin_ids(coin_ids):
    """
    Encodes the given coin IDs using LabelEncoder.

    Args:
    - coin_ids (list or ndarray): Array-like object containing coin IDs.

    Returns:
    - coin_ids_encoded (ndarray): Encoded coin IDs, reshaped to (-1, 1).
    - label_encoder (LabelEncoder): The fitted LabelEncoder object.
    """
    # Initialize and fit the LabelEncoder
    label_encoder = LabelEncoder()
    coin_ids_encoded = label_encoder.fit_transform(coin_ids).reshape(-1, 1)

    return coin_ids_encoded, label_encoder


# LSTM_ model feature engineering

"""

Preparing the training and validation sequence for the LSTM_ for 30 day context window

"""
def create_sequences_and_targets(data, feature_columns, high_column, low_column, sequence_length=30):
    """
    Creates sequences and corresponding target values from the data.

    Args:
    - data (DataFrame): DataFrame containing the historical data, features, and targets.
    - feature_columns (list): List of column names for features used in sequences.
    - high_column (str): Column name for % Diff From High target.
    - low_column (str): Column name for % Diff From Low target.
    - sequence_length (int): Length of each sequence (default is 30 days).

    Returns:
    - sequences (ndarray): Array of shape (num_sequences, sequence_length, num_features).
    - targets (ndarray): Array of shape (num_sequences, 2), aligned with sequences.
    """
    sequences = []
    targets = []

    # Loop over data to create sequences and corresponding targets
    for coin, group in data.groupby('coin_index'):
        group = group.sort_values(by='date')  # Ensure data is sorted by date

        # Generate sequences for each coin
        for i in range(len(group) - sequence_length + 1):
            sequence = data[feature_columns].iloc[i:i + sequence_length].values
            sequences.append(sequence)

        # Extract target values corresponding to the end of the sequence
            target = data[[high_column, low_column]].iloc[i + sequence_length - 1].values
            targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    return sequences, targets

# Example feature columns and target columns

# variable1 = '7'
# variable2 = '5'

# feature_columns = [f'Days_Since_High_Last_{variable1}_Days', f'%_Diff_From_High_Last_{variable1}_Days',
#                    f'Days_Since_Low_Last_{variable1}_Days', f'%_Diff_From_Low_Last_{variable1}_Days']
# high_column = f'%_Diff_From_High_Next_{variable2}_Days'
# low_column = f'%_Diff_From_Low_Next_{variable2}_Days'

# # Generate sequences and targets
# sequences, targets = create_sequences_and_targets(df_input, feature_columns, high_column, low_column)

# print("Sequences shape:", sequences.shape)  # Expected: (num_sequences, 30, 4)
# print("Targets shape:", targets.shape)      # Expected: (num_sequences, 2)

"""
# TRAIN_TEST_SPLIT_FUNCTION

"""
# TRAIN_TEST_SPLIT_FUNCTION


def train_test_split_sequences(sequences, coin_ids_encoded, targets, test_size=0.2, random_state=42):
    """
    Splits the sequences, coin IDs, and targets into training and validation sets.

    Args:
    - sequences (ndarray): Array of shape (num_samples, sequence_length, num_features) with feature sequences.
    - coin_ids_encoded (ndarray): Encoded coin IDs, of shape (num_samples, 1).
    - targets (ndarray): Target values, of shape (num_samples, 2), containing the two target variables.
    - test_size (float): Fraction of data to be used for validation.
    - random_state (int): Seed for reproducibility of the split.

    Returns:
    - X_train_seq, X_val_seq: Training and validation sequences.
    - X_train_coin, X_val_coin: Training and validation coin IDs.
    - y_train, y_val: Training and validation targets.
    """
    # Split the sequences and encoded coin IDs
    X_train_seq, X_val_seq, X_train_coin, X_val_coin = train_test_split(
        sequences, coin_ids_encoded, test_size=test_size, random_state=random_state
    )

    # Split the targets
    y_train, y_val = train_test_split(targets, test_size=test_size, random_state=random_state)

    return X_train_seq, X_val_seq, X_train_coin, X_val_coin, y_train, y_val

