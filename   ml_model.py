import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




def train_model(sequence_length, num_features, num_coins, embedding_dim=8, lstm_units=64, dropout_rate=0.3):
    """
    Builds an LSTM model with an embedding layer for coin IDs.

    Args:
    - sequence_length (int): Length of each input sequence.
    - num_features (int): Number of features in each time step.
    - num_coins (int): Number of unique coins (for embedding).
    - embedding_dim (int): Dimension of the embedding space for coin IDs.
    - lstm_units (int): Number of units in the LSTM layer.
    - dropout_rate (float): Dropout rate for regularization.

    Returns:
    - model (tf.keras.Model): Compiled LSTM model.
    """
    # Input for the sequence data
    sequence_input = Input(shape=(sequence_length, num_features), name="sequence_input")
    x = LSTM(lstm_units, return_sequences=False)(sequence_input)
    x = Dropout(dropout_rate)(x)

    # Input for the encoded coin IDs
    coin_id_input = Input(shape=(1,), name="coin_id_input")
    coin_embedding = Embedding(input_dim=num_coins, output_dim=embedding_dim, input_length=1)(coin_id_input)
    coin_embedding = Flatten()(coin_embedding)  # Flatten embedding to 1D

    # Concatenate LSTM output with coin embedding
    combined = Concatenate()([x, coin_embedding])

    # Dense layers after concatenation
    dense_output = Dense(64, activation="relu")(combined)
    dense_output = Dropout(dropout_rate)(dense_output)
    dense_output = Dense(32, activation="relu")(dense_output)

    # Final output layer with 2 units for the two target values
    output = Dense(2, activation="linear", name="output")(dense_output)

    # Define and compile the model
    model = Model(inputs=[sequence_input, coin_id_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


# make the plotting function 


def plot_training_history(history):
    # Plot training and validation loss
    plt.figure(figsize=(14, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

# prediction function 

import numpy as np

def predict_next_diff(model, sequences, coin_ids_encoded):
    """
    Make predictions using the trained LSTM model.

    Args:
    - model (tf.keras.Model): The trained LSTM model.
    - sequences (np.ndarray): The input sequences of shape (n_samples, 30, num_features).
    - coin_ids_encoded (np.ndarray): The encoded coin IDs of shape (n_samples, 1).

    Returns:
    - predictions (np.ndarray): The predicted % differences from high and low for the next N days.
    """
    # Ensure the input data is in the correct shape
    sequences = np.array(sequences)
    coin_ids_encoded = np.array(coin_ids_encoded)

    # Make predictions
    predictions = model.predict([sequences, coin_ids_encoded])

    return predictions