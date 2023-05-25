from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
import numpy as np
from time import sleep

def main():
    env = StarCraft2Env(map_name="8m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10
    
    observations, state = env.reset()
    print(len(observations), observations[0].shape, state.shape)

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            env.render()  # Uncomment for rendering
            
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                #print(avail_actions, avail_actions_ind)
                
                
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
                #print(agent_id, action)

            reward, terminated, _ = env.step(actions)
            #print(reward)
            episode_reward += reward
            #sleep(0.1)


        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()
