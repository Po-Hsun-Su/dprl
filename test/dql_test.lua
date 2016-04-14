--[[ 
test bench for dql

]]--
require 'rPrint'
require 'optim'
local dprl = require 'init'

-- initialize environment
local RandomWalk = require 'rlenvs.RandomWalk'
local env = RandomWalk()
local stateSpec = env:getStateSpec()
local actionSpec = env:getActionSpec()
print('stateSpec')
rPrint(stateSpec)
print('actionSpec')
rPrint(actionSpec)
-- construct Q network (Note that the Q network must operate in minibatch mode)
require 'nn'
require 'dpnn'
local qnet = nn.Sequential() 
  -- input of qnet is N x D where N is the size of minibatch and D is the size of state
  -- Because state space is discrete, we collapse the input to a 1D vector of size ND for the following OneHot module.   
qnet:add(nn.Collapse(2))

  -- Convert to one hot coding. 
local stateRange = stateSpec[3][2] - stateSpec[3][1] + 1 -- see rlenvs
qnet:add(nn.OneHot(stateRange)) -- The dimension of output is ND x <stateRange> 
--print(qnet:forward(torch.Tensor{{1},{2},{3},{4}})) -- test

  -- Add hidden layers
local hiddenSize = 8
local transfer = 'ReLU'
qnet:add(nn.Linear(stateRange, hiddenSize))
qnet:add(nn[transfer]())
--print(qnet:forward(torch.Tensor{{1},{2},{3},{4}})) -- test

  -- Output Q values
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1
qnet:add(nn.Linear(hiddenSize,actionRange))
--print(qnet:forward(torch.Tensor{{1},{2},{3},{4}})) -- test

-- initialize dqn
local dqn_param = {capacity = 32, batchSize = 4, discount = 0.9, epslon = 0.1}
local optimConfig = {learningRate = 0.01,
                     momentum = 0.0}
local optimMethod = optim.rmsprop
local dqn = dprl.dqn(qnet,dqn_param, optimMethod, optimConfig)
-- initialize dql
local dql_param = {step = 20, lr = 0.1, updateInterval = 8}
local preprop = function (observation)
                  return torch.Tensor{observation + 1}
                end
local oneHot2ID = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actPreprop = function (action)
                      return action*oneHot2ID
                    end
local dql = dprl.dql(dqn, env, dql_param, preprop, actPreprop)

local report = function(dql)
  print(dql.dqn.Qvalue)
end



dql:learning(5000,report)


