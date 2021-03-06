# Entity-oriented PPO Demo

This is a demo of **"entity-oriented" deep reinforcement learning** (name is inspired by "object-oriented programming"). 

Feeling frustrated about not being able to describe your game's observation and action space  using `gym.spaces.Discrete` or `gym.spaces.Discrete` for RL libraries?  Entity-oriented RL comes to the rescue - it gives a much more intuitive observation and action space API to game practioners. Specifically, Entity-oriented RL allows game practioners to describe the entities in the game using JSON-like syntax such as follows:

https://github.com/vwxyzjn/entity-ppo-demo/blob/f8809e172e141880c84007b6e809c3e93b714118/main.py#L40-L63

The `main.py` contains an end-to-end example. Please follow the steps below to get started.

## Colab

To help get started, we have prepared a colab notebook here - https://colab.research.google.com/drive/1WM23R9TA-C6rmDTFAy4hbnB87C1hcmM2?usp=sharing#scrollTo=HGzNOrnSiaHF

## Local Setup

```bash
poetry install
poetry run pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

## Run an experiment

```bash
poetry run python main.py total_timesteps=1000000
```

The schema and documentation of configuration flags (such as `total_timesteps`) can be found [here](https://github.com/entity-neural-network/enn-trainer/blob/main/config-schema.ron)

## Visualize metrics

```bash
tensorboard --logdir runs
```

![Demo](demo.png)

## More info

To learn more about how to formulate observation space and action space with `entity-gym`, see https://entity-gym.readthedocs.io/en/latest/quick-start-guide.html.

For more advanced training examples with more complex games, see https://github.com/entity-neural-network/enn-zoo
