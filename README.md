## dprl
Deep reinforcement learning package for torch7. 

Supported algorithms:
* dprl.dql: Deep Q learning [[1]](#references)

To be supported algorithms:
* dprl.ddpg: Deep deterministic policy gradient [[2]](#references)

## Installation

```
git clone https://github.com/PoHsunSu/dprl.git
cd dprl
luarocks make dprl-scm-1.rockspec
```
## Usage
#### Deep Q learning (dprl.dql)
dprl.dql implements algorithm 1 in [[1]](#references).

1. Initialize a deep Q learning agent.
	```
	local dql = dprl.dql(dqn, env, config, statePreprop, actPreprop)
	```

	parameters:

	* `dqn`: a deep Q network. See [dprl.dqn](#dqn) below.

	* `env`: an eviroment with interfaces following [rlenvs](https://github.com/Kaixhin/rlenvs#api).

	* `config`: a table containing configurations of `dql`
		* step: number of steps which an episode terminates
		* updatePeriod: number of steps between successive updates of target network

	* `statePreprop` (optional): a function processing observation from `env` into state for `dqn`. For example, observation is number and we need to convert it to tensor:
		```
		local statePreprop = function (observation)
			return torch.Tensor(observation)
		end
		```

	* `actPreprop` (optional): a function processing output of `dqn` into action for `env`. For example, `dqn` outputs action in onehot coding and we need to convert it to number index:
		```
		local actPreprop = function (action)
			return action*torch.linspace(1, action:size(1), action:size(1))
		end  
		```

2. Learning
	```
	dql:learning(episode, report)
	```
	parameters:
	* `episode`: number of training episodes
	* `report` (optional): a function called at the end of each episode for reporting the status of training.

3. Testing
	```
	dql:testing(episode, visualization)
	```
	parameters:
	* `episode`: number of testing episodes
	* `visualization` (optional): a function called at the end of each step for visualization of `dqn`.

####<a name="dqn"></a>Deep Q network (dprl.dqn)
`dprl.dqn` implements experience replay and trains the neural network in itself.

1. Initialize `dprl.dqn`
	```
	local dqn = dprl.dqn(qnet, param, optim, optimConfig)
	```

	parameters:
	* `qnet`: a neural network model built with the [nn](https://github.com/torch/nn) package. It estimates the values of all actions given input state.
	* `config`: a table containing the configurations of `dqn`
		* `replaySize`: size of replay memory	
		* `batchSize`: size of mini-batch of training cases sampled on each replay
		* `discount`: discount factor of reward 	
		* `epsilon`: the ε of ε-greedy exploration
	* `optim`: optimization in the [optim](https://github.com/torch/optim) package for training `qnet`. 
	* `optimConfig`: configuration of `optim`

##<a name="API"></a>API


## References
[1] Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature 518(7540): 529-533.
[2] Lillicrap, T. P., et al. (2015). "Continuous control with deep reinforcement learning." CoRR abs/1509.02971.
<!---
## TODO
#### dqn, dql

- [x] Add test scripts of using optim
- [x] Implement remaining mechenics of DQN
- [x] Finish readme

- [ ] Cuda support
- [ ] Prioritized experience replay
-->
