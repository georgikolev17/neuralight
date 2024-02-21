import torch
import torch.nn as nn


class QlAgent:
    def __init__(self, input_shape=165, output_shape=2**15, loss_fn=nn.MSELoss):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )
        self.loss_fn = loss_fn

    def take_action(self, observations):
        if observations.shape() != (self.input_shape, ):
            raise Exception(f'Invalid input shape. Expected ${self.input_shape}, got ${observations.shape()}!')
        pred_rewards = self.model(observations)

    def learn(self, pred_reward, actual_reward):
        self.loss_fn(pred_reward, actual_reward)





# Define input and output sizes
input_Size = 33*5
output_Size = 2**15

# Initialize Q-network
q_network = QlAgent(input_Size, output_Size)

# Print the network architecture
print(q_network)
