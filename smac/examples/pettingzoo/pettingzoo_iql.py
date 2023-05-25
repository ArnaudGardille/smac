from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from time import sleep

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from tqdm import trange

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from cleanrl_utils.evals.dqn_eval import evaluate
from cleanrl_utils.huggingface import push_to_hub

from smac.env.pettingzoo import StarCraft2PZEnv
from torch.utils.tensorboard import SummaryWriter

import stable_baselines3 as sb3

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--use-state", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether we give the global state to agents instead of their respective observation")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=10,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=1,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=11,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=1,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, obs_shape, act_shape):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_shape, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, act_shape),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

    

class QAgent():
    def __init__(self, env, agent_id, args, obs_shape, act_shape):
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.agent_id = agent_id
        self.action_space = env.action_space(agent_id)

        self.q_network = QNetwork(env, obs_shape, act_shape).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network = QNetwork(env, obs_shape, act_shape).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        #print(self.buffer_size,env.observation_space(self.agent_id),env.action_space(self.agent_id),device)
        self.rb = DictReplayBuffer(
            self.buffer_size,
            env.observation_space(self.agent_id), #['observation'],
            env.action_space(self.agent_id),
            device,handle_timeout_termination=False,
            )

        self.obs = None

    def act(self, dict_obs, global_step=0):
        obs, avail_actions = dict_obs['observation'], dict_obs['action_mask']
        avail_actions_ind = np.nonzero(avail_actions)[0]
        
        epsilon = linear_schedule(self.start_e, self.end_e, self.exploration_fraction * self.total_timesteps, global_step)

        if random.random() < epsilon:
            actions = np.random.choice(avail_actions_ind)
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            considered_q_values = q_values*avail_actions
            actions = torch.argmax(considered_q_values, dim=1).cpu().numpy()

        avail_actions_ind = np.nonzero(avail_actions)[0]
        assert actions in avail_actions_ind

        return actions

    def train(self, global_step):
        # ALGO LOGIC: training.
        if global_step > self.learning_starts:

            if global_step % self.train_frequency == 0:
                print('on train!', self.rb.buffer_size, self.rb.full)
                data = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    target_max, _ = self.target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
                old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 1 == 0:
                    self.writer.add_scalar("losses/td_loss", loss, global_step)
                    self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update target network
            if global_step % self.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                    )

        if self.save_model:
            self.save()

        

        if self.upload_model:
            self.upload_model()

    def add_to_rb(self, next_obs, actions, reward, terminated, truncated, infos):
        
        if self.obs is not None:
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            if truncated:
                real_next_obs = infos["final_observation"]
            self.rb.add(self.obs, real_next_obs, actions, reward, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        self.obs = next_obs

    def save(self):

        model_path = f"runs/{run_name}/{self.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            self.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        
    def upload_model(self):

        repo_name = f"{self.env_id}-{self.exp_name}-seed{self.seed}"
        repo_id = f"{self.hf_entity}/{repo_name}" if self.hf_entity else repo_name
        push_to_hub(self, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")


    



def main():
    
    ### Fioriture

    args = parse_args()
    run_name = f"iql_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    ### Creating Env
    env = StarCraft2PZEnv.env(map_name="8m")
    env.reset()

    agent_0 = env.agents[0]
    nb_obs = int(env.observation_space(agent_0)['observation'].shape[0])
    nb_act = int(env.action_space(agent_0).n)
    
    print('-'*20)
    print('agents: ',env.agents)
    print('num_agents: ',env.num_agents)
    print('observation_space: ',env.observation_space(agent_0))
    print('action_space: ',env.action_space(agent_0))
    print('infos: ',env.infos)    
    print('nb_obs: ',nb_obs)    
    print('nb_act: ',nb_act)    
    print('-'*20)
    
    ### Creating Agents
    q_agents = {agent:QAgent(env, agent, args, nb_obs, nb_act)  for agent in env.agents}

    start_time = time.time()

    total_reward = 0
    done = False
    completed_episodes = 0

    for completed_episodes in trange(args.total_timesteps):
        env.reset()
        for agent_id in env.agent_iter():
            #print("agent: ", agent_id)
            env.render()

            obs, reward, terms, truncs, infos = env.last()
            total_reward += reward



            if terms or truncs:
                action = None
            else:
                assert isinstance(obs, dict) and "action_mask" in obs
                #action = random.choice(np.flatnonzero(obs["action_mask"]))
                action = q_agents[agent_id].act(obs)

                q_agents[agent_id].add_to_rb(obs, action, reward, terms, truncs, infos)
            
            q_agents[agent_id].train(completed_episodes)

            env.step(action)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)


    env.close()

    print("Average total reward", total_reward / args.total_timesteps)


if __name__ == "__main__":
    main()
