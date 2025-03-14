# Design of the code

## The Agent

The `agent` is only responsible for **training**, and the parameter update subroutine is included in **update**.
For most algorithms, one epoch of training is described as:
1. Sample a sequence of trajectories by `rollout`, the trajectories are stored in `buffer` of current agent.
2. Update the parameters of `model` with given algorithm, like `REINFORCE`, `A2C`, `PPO`, etc.
3. Record training info. 

## The Model

The `model` takes care of the observation input and action output, including transformation etc. in `forward` specifically.
It should also implement a `select_action` method to select an action given current observation.

If you want to support more kinds of spaces, you should modify the model.
In particular, the `forward` function and the `select_action` function.