--[[ 
test bench for dql

]]--
require 'rPrint'
require 'optim'
require 'nn'
require 'dpnn'
require 'Bootstrap'

local dprl = require 'init'

-- initialize environment
local DeepChain = require 'rlenvs.DeepChain'
local env = DeepChain({length = 25})
local stateSpec = env:getStateSpec()
local actionSpec = env:getActionSpec()
print('stateSpec')
rPrint(stateSpec)
print('actionSpec')
rPrint(actionSpec)
-- construct Q network (Note that the Q network must operate in minibatch mode)

local qnet = nn.Sequential() 

local stateRange = stateSpec[3][2] - stateSpec[3][1] + 1 -- see rlenvs
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1
local tabular = nn.LookupTable(stateRange,actionRange)

local headNum = 20
local param_init = 1
qnet:add(nn.View(-1))
qnet:add(nn.Bootstrap(tabular, headNum, param_init))
--print(qnet:forward(torch.Tensor{{1},{2},{3},{4}})) -- test

-- initialize dqn
local bdqn_param = {headNum = headNum, replaySize = 200, batchSize = 16, discount = 1, epsilon = 0.1}
local optimConfig = {learningRate = 0.01,
                     momentum = 0.9}
local optimMethod = optim.nag
local bdqn = dprl.bdqn(qnet,bdqn_param, optimMethod, optimConfig)
-- initialize dql
local dql_param = {step = 200, updatePeriod = 16}
local preprop = function (observation)
                  return torch.Tensor{observation}
                end
local oneHot2ID = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actPreprop = function (action)
                      return action*oneHot2ID
                    end
local bdql = dprl.bdql(bdqn, env, dql_param, preprop, actPreprop)

local report = function(bdql,totalReward, active)
  print('totalReward', totalReward, 'active', active)
end

optimConfig.learningRate = 0
bdql:learning(100,report)
print('----initial explore ended------')
io.read()
optimConfig.learningRate = 0.01
bdql:learning(2000,report)

