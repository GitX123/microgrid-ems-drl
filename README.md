# Microgrid Energy Management using Deep Reinforcment Learning
This repository contains experiment code for my master thesis "Time Series Observation and Action Handling for Battery Management in Applying Deep Reinforcement Learning for
Microgrid Energy Management".

# Abstract
Time Series Observation and Action Handling for Battery Management in Applying Deep Reinforcement Learning for Microgrid Energy Management / 
The transformation from traditional grids to microgrids introduces challenges due to multiple distributed energy resources and the intermittency of renewable energy sources and loads. Much effort has been committed to the design of microgrid energy management systems to attain optimal operation, and reinforcement learning is considered one of the most promising methods because of its competitive properties. Reinforcement learning algorithms generally do not assume precise models and can learn the underlying dynamics of the system under uncertainty by interacting with the environment. However, directly applying reinforcement learning to microgrid energy management is not an easy task. In this paper, we study two design aspects in reinforcement learning algorithms for microgrid energy management, which are related to time series observation and battery management in microgrids. In order to process time series data and handle varying battery charging/discharging bounds in our deep reinforcement learning algorithm, recurrent neural networks and valid action space mapping are used in our implementation. Experimental results confirm that the two design aspects are crucial for applying reinforcement learning in microgrid energy management.

# Code Explanation
| File                  | Description         |
|-----------------------|---------------------|
| cigre_mv_microgrid.py | Contains code for creating our test grid               |
| data.py               | Convert data from PJM for our environment|
| main.py               | Entry point of our experiment |
| setting.py            | Environment settings                 |
| utils.py              | Some frequently used repeated functions           |

|Directory   |Description   |
|---|---|
| controllers  | Controllers for microgrid energy management using various algorithms  |
| data   | Processed data for our environment  |
|history| Training history |
|model_weights| Trained model weights|
|pf_res| Results of power flow analysis|
|plot| Plots of experimental results |
|rms| Store values for input normalization and running mean std|

## main.py
- train_ppo(): train PPO agent.
- train_td3(): train TD3 agent
- test(): test with the trained agent.
- baseline(): test baseline.
