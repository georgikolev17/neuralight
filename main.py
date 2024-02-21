# nets/grid4x4/grid4x4.net.xml
import re
import sumo_rl
import random
import numpy as np

env = sumo_rl.SumoEnvironment(net_file='nets/grid4x4/grid4x4.net.xml',
                  route_file='nets/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600,
                  single_agent=False)
observations = env.reset()
epochs = 500

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
print(neighbours)
# for i in range(epochs):
#     actions = {agent: random.randint(0, 7) for agent in env.traffic_signals}  # this is where you would insert your policy
#     observations, rewards, done, infos = env.step(actions)
#
#     print('-----------------')
#     # print(np.average(np.array(list(rewards.values()))))
