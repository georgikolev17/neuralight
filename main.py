# nets/grid4x4/grid4x4.net.xml

import sumo_rl
import numpy as np
env = sumo_rl.parallel_env(net_file='nets/grid4x4/grid4x4.net.xml',
                  route_file='nets/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600)
observations = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(env.action_space('B1'))
    # print(actions)
    print(observations.items())

    print('-----------------')
    # print(np.average(np.array(list(rewards.values()))))
