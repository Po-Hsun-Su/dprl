require 'nn'
require 'dpnn'
require 'rPrint'
require 'pprint'
require 'optim'
require 'rPrintModule'
require 'EntropyRegularization'

local dprl = require 'init'

-- initialize env
local rlenvs = require 'rlenvs'
local env = rlenvs.WindyWorld()
local stateSpec = env:getStateSpec()
local actionSpec = env:getActionSpec()
print('stateSpec')
rPrint(stateSpec)
print('actionSpec')
rPrint(actionSpec)

-- model
local stateRangeX = stateSpec[1][3][2] - stateSpec[1][3][1] + 1
local stateRangeY = stateSpec[2][3][2] - stateSpec[2][3][1] + 1
local stateRange = stateRangeX*stateRangeY
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1
local cnet = nn.Sequential()
cnet:add(nn.View(-1))
cnet:add(nn.LookupTable(stateRange,1))

--print('cnet:forward', cnet:forward(torch.Tensor{{1}}))
local beta = 0.01
local stochastic = false
local anet =  nn.Sequential()
anet:add(nn.View(-1))
anet:add(nn.LookupTable(stateRange,actionRange))
--anet:add(nn.rPrintModule('After LookupTable'))
anet:add(nn.SoftMax())
--anet:add(nn.rPrintModule('After SoftMax'))
anet:add(nn.EntropyRegularization(beta))
--anet:add(nn.rPrintModule('After EntropyRegularization'))
anet:add(nn.ReinforceCategorical(stochastic))

--print('anet:forward', anet:forward(torch.Tensor{{1}}))

-- initialize aac
local config = {tmax = 5, discount = 1}
local optimMethod = optim.rmsprop
local optimConfig = {learningRate = 0.01,
                     alpha = 0.95,
                     epsilon = 1e-8}
local aac = dprl.aac(anet, cnet, config, optimMethod,optimConfig)

-- initialize async
local loadPackage = function(threadIdx)
  require 'rPrint'
  require 'rPrintModule'
  require 'EntropyRegularization'
  -- mask rPrint except thread 1
  if threadIdx ~= 1 then
    rPrint = function () end
    print = function () end
  end
end
local asyncConfig = {nthread = 4, loadPackage = loadPackage, maxSteps = 10000}
local statePreprop =  function(observation)
  
  return torch.Tensor{(observation[2]-1)*stateRangeX + observation[1]}
end
local oneHot2Index = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actionPreprop =  function(action)
  local act = action*oneHot2Index
  return act[1]
end



local a3c = dprl.async(aac, env, asyncConfig, statePreprop, actionPreprop)
--print('Before learning: a3c.sharedAgent:getParameters()')
--rPrint(a3c.sharedAgent:getParameters())

--print('Before learning: a3c.sharedAgent:getOptimState()')
--rPrint(a3c.sharedAgent:getOptimState())
local totalReward = 0
local learningReport = function(trans)
  --print('learning report')
  totalReward = totalReward + trans.r
  --rPrint(trans)
  if trans.t then
    print('totalReward at traning', totalReward)
    totalReward = 0
  end
end

a3c:learn(100000, learningReport)
--print('After learninga3c.sharedAgent:getParameters()')
--rPrint(a3c.sharedAgent:getParameters()[2]:view(10,7))

--print('After learninga3c.sharedAgent:getOptimState()')
--rPrint(a3c.sharedAgent:getOptimState())
local totalReward = 0


local oneHot2Index = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actionPreprop =  function(action)
  local p, id = torch.max(action:view(-1),1)
  --print(id)
  --print(oneHot2Index[id[1]])
  return oneHot2Index[id[1]]
end

local report = function(s, a, r, ns, t)
  --a = actionPreprop(a)
  --print('s = ', s, 'a = ', a, 'r = ', r)
  totalReward = totalReward + r
  if t then
    print(totalReward)
    totalReward = 0
  end
end
a3c:test(30,report, actionPreprop)






