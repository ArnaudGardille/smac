

# %%
from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()
api_test(env, num_cycles=1000, verbose_progress=False)