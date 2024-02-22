import random

import torch
import torch.nn as nn


class QlAgent:
    def __init__(self, input_shape=165, output_shape=8, loss_fn=nn.MSELoss, learning_rate=1e-03, epsilon=1):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def predict_rewards(self, observations):
        if observations.shape != (self.input_shape, ):
            raise Exception(f'Invalid input shape. Expected ${self.input_shape}, got ${observations.shape()}!')
        pred_rewards = self.model(observations)
        return pred_rewards

    def learn(self, pred_reward, actual_reward):
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred_reward, actual_reward)
        loss.backward()
        self.optimizer.step()





# Define input and output sizes
input_Size = 33*5
output_Size = 2**15

# Initialize Q-network
q_network = QlAgent(input_Size, output_Size)

# Print the network architecture
print(q_network)
