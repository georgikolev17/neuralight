# nets/grid4x4/grid4x4.net.xml
import re

import numpy
import sumo_rl
import random
import numpy as np
import torch

from agents.ql_agent import QlAgent

env = sumo_rl.SumoEnvironment(net_file='nets/grid4x4/grid4x4.net.xml',
                  route_file='nets/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600,
                  single_agent=False)
observations = env.reset()

pattern1 = re.compile(r"[A-Z]\d[A-Z]\d")  # Capital letter, number, capital letter, number
print(env.sumo.trafficlight.getIDList())
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
print(neighbours)

epochs = 5
epsilon=1
gamma=0.9

ql_agents = {
    ts: QlAgent()
    for ts in env.ts_ids
}

for i in range(epochs):
    observations = env.reset()
    done = {"__all__": False}
    i=0
    while not done["__all__"]:
        input_data = {}
        for ts in observations:
            concatenated_array = np.concatenate([observations[n] for n in neighbours[ts]] + [observations[ts]])
            if len(concatenated_array) < 165:
                padded_array = np.concatenate((concatenated_array, np.full(165 - len(concatenated_array), -1)))
            else:
                padded_array = concatenated_array[:165]
            input_data[ts] = torch.tensor(padded_array).to(torch.float)

        print(input_data['A1'].shape)
        print(len(input_data['A1']))
        pred_rewards = {ts: ql_agents[ts].predict_rewards(input_data[ts]) for ts in env.traffic_signals}
        print(pred_rewards)
        actions = {
            agent:
                random.randint(0, 7) if random.random() < epsilon
                else torch.argmax(pred_rewards[agent], dim=1).item()
            for agent in env.traffic_signals
        }
        print(actions)
        observations, rewards, done, infos = env.step(actions)
    # print(np.average(np.array(list(rewards.values()))))
