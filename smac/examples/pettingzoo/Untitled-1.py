import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--i', type=int, nargs='?', default=None,
                    help='an integer for the accumulator')

args = parser.parse_args()
print(args.i is None)

def test():
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    env = tictactoe_v3.env()
    env.reset(seed=0)

    #env = gym.make('CartPole-v1')
    #env = gym.wrappers.RecordEpisodeStatistics(env)

    size_obs = 3*3*2
    size_act = 8

    #agent = QAgent(env, 'agent', 0, args, size_obs, size_act)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    agent_0 = env.agents[0]
    if args.use_state:
        size_obs = int(env.state_space['observation'].shape[0])
    else:
        size_obs = int(env.observation_space(agent_0)['observation'].shape[0])
    
    if args.add_id:
        size_obs += 8
    
    size_act = int(env.action_space(agent_0).n)
    
    
    print('-'*20)
    print('agents: ',env.agents)
    print('num_agents: ',env.num_agents)
    print('observation_space: ',env.observation_space(agent_0))
    print('action_space: ',env.action_space(agent_0))
    #print('infos: ',env.infos)    
    print('size_obs: ',size_obs)    
    print('size_act: ',size_act)    
    print('-'*20)
    
    print('agents:', env.agents)
    ### Creating Agents
    
    enc = OneHotEncoder(sparse_output=False).fit(np.array(env.agents).reshape(-1, 1))
    one_hot = {agent:enc.transform(np.array([agent]).reshape(-1, 1))[0] for agent in env.agents}

    q_agents = {agent:QAgent(env, agent, i, args, size_obs, size_act, one_hot[agent])  for i, agent in enumerate(env.agents)}

    if args.single_agent:
        agent_0 = q_agents[env.agents[0]]
        for agent in q_agents:
            q_agents[agent].q_network = agent_0.q_network
            q_agents[agent].replay_buffer = agent_0.replay_buffer

       

    if False:
        for agent in q_agents.values():
            print()
            print('-'*20)
            print(agent)
            print('-'*20)

    #total_reward = 0
    completed_episodes = 0

    pbar=trange(args.total_timesteps)
    for completed_episodes in pbar:
        env.reset()
        obs, _, _, _, _ = env.last()
        if args.use_state:
            obs['observation'] = env.state()

        episodic_returns = {}
        nb_steps = 0

        while env.agents:
        #for agent_id in env.agent_iter():
            if args.display_video:
                env.render()
            
            if args.use_state:
                for a in env.agents:
                    obs[a]['observation'] = env.state()

            
                
            actions = {agent: q_agents[agent].act(obs[agent], completed_episodes) for agent in env.agents}  

            #actions = {agent: np.random.choice(np.nonzero(obs[agent]['action_mask'])[0]) for agent in env.agents}  

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            if args.use_state:
                next_obs['observation'] = env.state()
                print(next_obs)

            for agent in next_obs:
                q_agents[agent].add_to_rb(obs[agent], actions[agent], rewards[agent], next_obs[agent], terminations[agent], truncations[agent], infos[agent])

            episodic_returns = {k: rewards.get(k, 0) + episodic_returns.get(k, 0) for k in set(rewards) | set(episodic_returns)}
            nb_steps += 1

            obs = next_obs

            #writer.add_scalars("Rewards", rewards, completed_episodes)
            #writer.add_scalar("Global Reward", global_reward, completed_episodes)

        if args.single_agent:
            agent_0.train(completed_episodes)
        else:
            for agent in q_agents.values():
                agent.train(completed_episodes)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalars("Episodic Return/", episodic_returns, completed_episodes)
        
        total_episodic_return = sum(episodic_returns.values())
        pbar.set_description(f"episodic return={total_episodic_return:5.1f}")
        writer.add_scalar("Total Episodic Return", total_episodic_return, completed_episodes)
        
        writer.add_scalar("Nb Steps", nb_steps, completed_episodes)


    env.close()

    #print("Average total reward", total_reward / args.total_timesteps)

