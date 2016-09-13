require 'nn'
require 'dpnn'
require 'rPrint'
require 'pprint'
require 'optim'
require 'rPrintModule'

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
local beta = 0.1
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
require 'dprl.rmsprop'
local optimMethod = optim.nag
local optimConfig = {learningRate = 0.001,
                     momentum = 0.9,
                     alpha = 0.95,
                     epsilon = 1e-8}
local aac = dprl.aac(anet, cnet, config, optimMethod,optimConfig)

-- initialize async
local loadPackage = function(threadIdx)
  require 'rPrint'
  require 'rPrintModule'
  require 'rlenvs'
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



local a3c = dprl.asyncl(aac, env, asyncConfig, statePreprop, actionPreprop)
--print('Before learning: a3c.sharedAgent:getParameters()')
--rPrint(a3c.sharedAgent:getParameters())

--print('Before learning: a3c.sharedAgent:getOptimState()')
--rPrint(a3c.sharedAgent:getOptimState())
local totalReward = 0
local learningReport = function(trans,t,T)
  --print('learning report')
  totalReward = totalReward + trans.r
  --rPrint(trans)
  if trans.t then
    print('totalReward at traning', totalReward)
    totalReward = 0
  end
end

a3c:learn(1000000, learningReport)
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

local testtotalReward = 0
local testEpisode = 30
local testStepReport = function(trans, t, E)
  testtotalReward = testtotalReward + trans.r
  if __threadid == 1 then
    xlua.progress(E, testEpisode)
  end
  local totalRewardCP = testtotalReward
  if trans.t then
    --print('testtotalReward',testtotalReward)
    testtotalReward = 0
  end
  return totalRewardCP
end

local averageTotalReward = 0
local testEpisodicReport = function(report, e)
  averageTotalReward = averageTotalReward + report
end

a3c:test(30,testStepReport, testEpisodicReport, actionPreprop)
print('Average total reward')
print(averageTotalReward/testEpisode)






