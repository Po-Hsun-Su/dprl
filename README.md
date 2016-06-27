## dprl
Deep reinforcement learning package for torch7. 

Algorithms:
* Deep Q-learning [[1]](#references)
* Double DQN [[2]](#references)
* Bootstrapped DQN (broken) [[3]](#references)
* Asynchronous advantage actor-critic [[4]](#references)

## Installation

```
git clone https://github.com/PoHsunSu/dprl.git
cd dprl
luarocks make dprl-scm-1.rockspec
```

## Example


## Library
The library provides implementation of deep reinforcement learning algorithms.

####<a name="dql"></a> dql
This class contains methods for Deep Q learning.
```
local dql = dprl.dql(dqn, env, config, statePreprop, actPreprop)
```

1. Initialize a deep Q learning agent.
	```
	local dql = dprl.dql(dqn, env, config, statePreprop, actPreprop)
	```

	arguments:

	* `dqn`: a deep Q-network or double DQN. See [dprl.dqn](#dqn) or [dprl.ddqn](#ddqn) below.

	* `env`: an eviroment with interfaces defined in [rlenvs](https://github.com/Kaixhin/rlenvs#api).

	* `config`: a table containing configurations of `dql`
		* step: number of steps before an episode terminates
		* updatePeriod: number of steps between successive updates of target Q-network

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
	arguments:
	* `episode`: number of training episodes
	* `report` (optional): a function called at the end of each episode for reporting the status of training.

3. Testing
	```
	dql:testing(episode, visualization)
	```
	arguments:
	* `episode`: number of testing episodes
	* `visualization` (optional): a function called at the end of each step for visualization of `dqn`.

####<a name="dqn"></a>Deep Q network (`dprl.dqn`)
`dprl.dqn` implements experience replay and training procedure of the neural network in itself. We only need to initialize it and give it to `dprl.dql`.

1. Initialize `dprl.dqn`
	```
	local dqn = dprl.dqn(qnet, config, optim, optimConfig)
	```

	arguments:
	* `qnet`: a neural network model built with the [nn](https://github.com/torch/nn) package. It estimates the values of all actions given input state.
	* `config`: a table containing the following configurations for `dqn`
		* `replaySize`: size of replay memory	
		* `batchSize`: size of mini-batch of training cases sampled on each replay
		* `discount`: discount factor of reward 	
		* `epsilon`: the ε of ε-greedy exploration
	* `optim`: optimization in the [optim](https://github.com/torch/optim) package for training `qnet`. 
	* `optimConfig`: configuration of `optim`

####<a name="ddqn"></a>Double DQN (`dprl.ddqn`)
Initilization of `dprl.ddqn` and `dprl.dqn` are identical. Double DQN is recommended because it prevents the overestimation problem in DQN [[2]](#references).


####<a name="bdql"></a>Bootstrapped deep Q-learning (dprl.bdql)
`dprl.bdql` implements learning procedure in Bootstrapped DQN. Except initialization, its usage is identical to `dprl.dql`.

1. Initialize a bootstrapped deep Q-learning agent. 

	```
	local bdql = dprl.bdql(bdqn, env, config, statePreprop, actPreprop)
	```
	Except the first arguments `bdqn`, which is an instance of [`dprl.bdqn`](#bdqn), definitions of the other arguments are the same in [`dprl.dql`](#dql).

####<a name="bdqn"></a>Bootstrapped deep Q-network (dprl.bdqn)
`dprl.bdqn` inherets [`dprl.dqn`](#dqn). It is customized for Bootstrapped Deep Q-network.

1. Initialize `dprl.bdqn`
	```
	local bdqn = dprl.bdqn(bqnet, config, optim, optimConfig)
	```
	
	arguments:
	* `bqnet`: a bootstrapped neural network with module [`Bootstrap`](#Bootstrap). 
	* `config`: a table containing the following configurations for `bdqn`
		* `replaySize`, `batchSize`, `discount` ,and `epsilon`: see `config` in [`dprl.dqn`](#dqn).	
		* `headNum`: the number of heads in bootstrapped neural network `bqnet`.
	* `optim`: see `optim` in [`dprl.dqn`](#dqn).
	* `optimConfig`: see `optimConfig` in [`dprl.dqn`](#dqn).

####<a name="asyncl"></a>Asynchronous learning (dprl.asyncl)
'dprl.asyncl'
####<a name="aac"></a>Avantage actor-critic (dprl.aac)

## <a name="mod"></a>Modules

####<a name="Bootstrap"></a>Bootstrap (`nn.Bootstrap`)
This module is for constructing bootstrapped network [[3]](#references). Let the shared network be `shareNet` and the head network be `headNet`. A bootstrapped network `bqnet` for `dprl.bdqn` can be constructed as follows:
```
require 'Bootstrap'

-- Definition of 'shareNet' and head 'headNet'


-- Decorate headNet with nn.Bootstrap
local boostrappedHeadNet = nn.Bootstrap(headNet, headNum, param_init)

-- Connect shareNet and boostrappedHeadNet
local bqnet = nn.Sequential():add(shareNet):add(boostrappedHeadNet)
```
`headNum`: the number of heads of the bootstrapped network

`param_init`: a scalar value controlling the range or variance of parameter initialization in headNet.
It is passed to method `headNet:reset(param_init)` after constructing clones of headNet.

##<a name="API"></a>API


## References
[1] Volodymyr Mnih et al., “Human-Level Control through Deep Reinforcement Learning,” Nature 518, no. 7540 (February 26, 2015): 529–33, doi:10.1038/nature14236.

[2] Hado van Hasselt, Arthur Guez, and David Silver, “Deep Reinforcement Learning with Double Q-Learning,” arXiv:1509.06461, September 22, 2015, http://arxiv.org/abs/1509.06461.

[3] Ian Osband et al., “Deep Exploration via Bootstrapped DQN,” arXiv:1602.04621, February 15, 2016, http://arxiv.org/abs/1602.04621.

[4] Volodymyr Mnih et al., “Asynchronous Methods for Deep Reinforcement Learning,” arXiv:1602.01783, February 4, 2016, http://arxiv.org/abs/1602.01783.


<!---
## TODO
#### dqn, dql

- [x] Add test scripts of using optim
- [x] Implement remaining mechenics of DQN
- [x] Finish readme
- [ ] Cuda support
- [ ] Double Q learning
- [ ] Deep exploration
- [ ] Prioritized experience replay
-->
