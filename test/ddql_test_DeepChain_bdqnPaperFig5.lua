--[[ 
test bench for dql

]]--
require 'rPrint'
require 'optim'
require 'nn'
require 'dpnn'
require 'Bootstrap'

math.randomseed( os.time() )

local dprl = require 'init'
local totalRewardRecord = {}
for length = 3, 25 do
  -- initialize environment
  local DeepChain = require 'rlenvs.DeepChain'
  local env = DeepChain({length = length})
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
  
  qnet:add(nn.View(-1))
  qnet:add(tabular)
  --print(qnet:forward(torch.Tensor{{1},{2},{3},{4}})) -- test
  
  -- initialize dqn
  local dqn_param = { replaySize = 1000, batchSize = 16, discount = 1, epsilon = 0.01}
  local optimConfig = {learningRate = 0.01,
                       momentum = 0.9}
  local optimMethod = optim.rmsprop
  local dqn = dprl.ddqn(qnet,dqn_param, optimMethod, optimConfig)
  -- initialize dql
  local dql_param = {step = 1000, lr = 0.1, updatePeriod = 30}
  local preprop = function (observation)
                    return torch.Tensor{observation}
                  end
  local oneHot2ID = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
  local actPreprop = function (action)
                        return action*oneHot2ID
                      end
  local bdql = dprl.dql(dqn, env, dql_param, preprop, actPreprop)
  totalRewardRecord[length] = {}
  local report = function(dql, totalReward)
    table.insert(totalRewardRecord[length],totalReward)
    print('totalReward', totalReward)
  end
  bdql:learning(2000,report)
end

torch.save('totalRewardRecord',totalRewardRecord)

