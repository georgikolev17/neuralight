# nets/grid4x4/grid4x4.net.xml
import re

import numpy
import sumo_rl
import random
import numpy as np
import torch
from matplotlib import pylab as plt

from agents.ql_agent import QlAgent

env = sumo_rl.SumoEnvironment(net_file='nets/grid4x4/grid4x4.net.xml',
                  route_file='nets/grid4x4/4x47500trips.rou.xml',
                  use_gui=True,
                  num_seconds=20000,
                  single_agent=False)
observations = env.reset()

pattern1 = re.compile(r"[A-Z]\d[A-Z]\d")  # Capital letter, number, capital letter, number
# Filter strings that match the patterns
edges = [s for s in env.sumo.edge.getIDList() if pattern1.fullmatch(s)]
neighbours = {}
for edge in edges:
    tl1 = edge[:(len(edge)-2)]
    tl2 = edge[(len(edge)-2):]
    if tl1 in neighbours and tl2 not in neighbours[tl1]:
        neighbours[tl1].append(tl2)
    elif tl1 not in neighbours:
        neighbours.update({tl1: [tl2]})
    if tl2 in neighbours and tl1 not in neighbours[tl2]:
        neighbours[tl2].append(tl1)
    elif tl2 not in neighbours:
        neighbours.update({tl2: [tl1]})
neighbours = {ts: numpy.array(neighbours[ts]) for ts in env.traffic_signals}

epochs = 5
epsilon = 1
gamma = 0.9

ql_agents = {
    ts: QlAgent()
    for ts in env.ts_ids
}


done = {"__all__": False}
i=0
avg_rewards = []
input_data = {}
for ts in observations:
    concatenated_array = np.concatenate([observations[ts] + observations[n] * 0.2 for n in neighbours[ts]])
    if len(concatenated_array) < 165:
        padded_array = np.concatenate((concatenated_array, np.full(165 - len(concatenated_array), -1)))
    else:
        padded_array = concatenated_array[:165]
    input_data[ts] = torch.Tensor(padded_array).to(torch.float)
while not done["__all__"]:
    i += 1
    pred_rewards = {ts: ql_agents[ts].predict_rewards(input_data[ts]) for ts in env.traffic_signals}
    actions = {
        agent:
            random.randint(0, 7) if random.random() < epsilon
            else torch.argmax(pred_rewards[agent], dim=0).item()
        for agent in env.traffic_signals
    }
    observations, rewards, done, infos = env.step(actions)

    input_data2 = {}
    for ts in observations:
        concatenated_array = np.concatenate([observations[ts] + observations[n] * 0.2 for n in neighbours[ts]])
        if len(concatenated_array) < 165:
            padded_array = np.concatenate((concatenated_array, np.full(165 - len(concatenated_array), -1)))
        else:
            padded_array = concatenated_array[:165]
        input_data2[ts] = torch.Tensor(padded_array).to(torch.float)
    curr_reward = rewards[ts]
    for ts in env.traffic_signals:
        for n in neighbours[ts]:
            rewards[ts] += rewards[n]*0.2
    with torch.no_grad():
        q_rewards = {ts: rewards[ts] + gamma * torch.max(ql_agents[ts].predict_rewards(input_data2[ts])) for ts in env.traffic_signals}
    avg_reward = sum(rewards.values())/len(rewards)
    avg_rewards.append(avg_reward)
    print(f'{i} {sum(rewards.values())/len(rewards)}')
    if i > 100:
        break
    for ts in env.traffic_signals:
        ql_agents[ts].learn(torch.Tensor(pred_rewards[ts][actions[ts]]), torch.Tensor([q_rewards[ts]]))
    if epsilon > 0.1:
        epsilon -= 1/1500
    input_data = input_data2
for agent in env.traffic_signals:
    torch.save(ql_agents[agent].model.state_dict(), f'trained_models/model{agent}.pth')
fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Epochs")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(9, 5)
ax.scatter(np.arange(len(avg_rewards)), avg_rewards)
plt.show()
observations = env.reset()
