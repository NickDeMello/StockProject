#--------------------------------------------------------
#----------------------DATA LOADER-----------------------
#--------------------------------------------------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Assuming your dataset has shape (3100, 11)
# X is the features, and y is the corresponding actions ("buy", "sell", "stay")

# Example data
num_samples = 3100
num_features = 11
sequence_length = 30

# Generate sample data (replace this with your actual dataset)
X = np.random.rand(num_samples, num_features)
y = np.random.choice(["buy", "sell", "stay"], size=num_samples)

# Label encoding for actions
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create sequences of states and corresponding actions
states = []
actions = []

for i in range(sequence_length, num_samples):
    sequence = X[i - sequence_length:i]  # Extract the past 30 instances as a sequence
    action = y_encoded[i]  # Action taken at the next time step

    states.append(sequence)
    actions.append(action)

# Convert lists to NumPy arrays
states = np.array(states)
actions = np.array(actions)

# Normalize or preprocess your data if needed

# Shuffle the data
indices = np.arange(states.shape[0])
np.random.shuffle(indices)

states = states[indices]
actions = actions[indices]

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(states))

train_states, test_states = states[:split_index], states[split_index:]
train_actions, test_actions = actions[:split_index], actions[split_index:]

# Now, you can use train_states, train_actions for training and test_states, test_actions for testing
