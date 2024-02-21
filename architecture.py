#-------------------------------------------
#---------------ARCHITECTURE----------------
#-------------------------------------------

# import tensorflow as tf
from tensorflow.keras import layers, models

# Assuming 30 samples, each with 11 features
input_shape_cnn = (30, 11, 1)

# CNN
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), dilation_rate=(2, 2), activation='relu', input_shape=input_shape_cnn, padding='same'))
cnn_model.add(layers.Conv2D(32, (3, 3), dilation_rate=(4, 4), activation='relu'))

# Print CNN model summary
cnn_model.summary()

# LSTM
lstm_model = models.Sequential()
lstm_model.add(layers.LSTM(156, activation='tanh', return_sequences=False))  # Set return_sequences to False for the final LSTM layer

# Print LSTM model summary
lstm_model.summary()

# MLP
mlp_model = models.Sequential()
mlp_model.add(layers.Dense(32, activation='elu'))
mlp_model.add(layers.Dense(3, activation='linear'))

# Print MLP model summary
mlp_model.summary()

# Create a Sequential model for the entire architecture
full_model = models.Sequential([cnn_model, lstm_model, mlp_model])
# Print the full model summary
full_model.summary()

