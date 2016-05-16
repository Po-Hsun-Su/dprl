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
for length = 10, 10 do
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
  
  local headNum = 20
  local param_init = 5
  qnet:add(nn.View(-1))
  qnet:add(nn.Bootstrap(tabular, headNum, param_init))
  --print(qnet:forward(torch.Tensor{{1},{2},{3},{4}})) -- test
  
  -- initialize dqn
  local bdqn_param = {headNum = headNum, replaySize = 400, batchSize = 16, discount = 1}
  local optimConfig = {learningRate = 0.01,
                       momentum = 0.9}
  local optimMethod = optim.nag
  local bdqn = dprl.bdqn(qnet,bdqn_param, optimMethod, optimConfig)
  -- initialize dql
  local dql_param = {step = 200, updatePeriod = 32}
  local preprop = function (observation)
                    return torch.Tensor{observation}
                  end
  local oneHot2ID = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
  local actPreprop = function (action)
                        return action*oneHot2ID
                      end
  local bdql = dprl.bdql(bdqn, env, dql_param, preprop, actPreprop)
  
  totalRewardRecord[length] = {}
  local report = function(bdql,totalReward, active, states, Qvalues)
    table.insert(totalRewardRecord[length],totalReward)
    local stateDist = {}
    for i = 1, #states do
      stateDist[states[i]] = (stateDist[states[i]] or 0) + 1
    end
    print('totalReward', totalReward, 'active', active)
    rPrint(stateDist)
    rPrint(Qvalues)
  end
  
  optimConfig.learningRate = 0
  bdql:learning(40,report)
  print('----initial explore ended------')
  optimConfig.learningRate = 0.01
  bdql:learning(1960,report)

end

torch.save('bdqn_totalRewardRecord',totalRewardRecord)

