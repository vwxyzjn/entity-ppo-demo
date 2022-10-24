# Entity-oriented PPO Demo

This is a demo of **"entity-oriented" deep reinforcement learning** (name is inspired by "object-oriented programming"). 

Feeling frustrated about not being able to describe your game's observation and action space  using `gym.spaces.Discrete` or `gym.spaces.Discrete` for RL libraries?  Entity-oriented RL comes to the rescue - it gives a much more intuitive observation and action space API to game practitioners. Specifically, Entity-oriented RL allows game practitioners to describe the entities in the game using JSON-like syntax such as follows:

https://github.com/vwxyzjn/entity-ppo-demo/blob/f8809e172e141880c84007b6e809c3e93b714118/main.py#L40-L63

The `main.py` contains an end-to-end example. Please follow the steps below to get started.

## Setup

```bash
poetry install
poetry run pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
```

## Run an experiment

```bash
poetry run python mine_sweeper_demo.py env.id=MineSweeper total_timesteps=1000000
xvfb-run -a poetry run python microrts_demo.py env.id=GymMicrorts rollout.num_envs=16 total_timesteps=1000000 rollout.steps=256 eval.capture_videos=True eval.interval=300000 eval.steps=2000 eval.num_envs=1 eval.processes=1
```

The schema and documentation of configuration flags (such as `total_timesteps`) can be found [here](https://github.com/entity-neural-network/enn-trainer/blob/main/config-schema.ron)

## Visualize metrics

```bash
tensorboard --logdir runs
```

![Demo](demo.png)


## Experiment tracking

```bash
poetry run python mine_sweeper_demo.py total_timesteps=1000000 track=true
xvfb-run -a poetry run python microrts_demo.py env.id=GymMicrorts rollout.num_envs=16 total_timesteps=1000000 rollout.steps=256 eval.capture_videos=True eval.interval=300000 eval.steps=2000 eval.num_envs=1 eval.processes=1 track=true
```

It is possible to track the experiments to weights and biases. To do so, please set `track=true` in the command line arguments. See the following tracked experiments:

* [Gym-microrts experiment](https://wandb.ai/entity-neural-network/enn-ppo/runs/3b49rbdu)
* [Mine Sweeper experiment](https://wandb.ai/entity-neural-network/enn-ppo/runs/2z2bd9os)

## More info

To learn more about how to formulate observation space and action space with `entity-gym`, see https://entity-gym.readthedocs.io/en/latest/quick-start-guide.html.

For more advanced training examples with more complex games, see https://github.com/entity-neural-network/enn-zoo
