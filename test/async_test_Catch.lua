require 'nn'
require 'dpnn'
require 'rPrint'
require 'pprint'
require 'optim'
require 'rPrintModule'
require 'EntropyRegularization'
require 'xlua'
torch.setnumthreads(1)
local dprl = require 'init'

-- initialize env
local rlenvs = require 'rlenvs'
local env = rlenvs.Catch()
local stateSpec = env:getStateSpec()
local actionSpec = env:getActionSpec()
print('stateSpec')
rPrint(stateSpec)
print('actionSpec')
rPrint(actionSpec)

-- action net
local beta = 0.01
local stochastic = true

local anet = nn.Sequential() 
local stateDim = stateSpec[2][1]*stateSpec[2][2]*stateSpec[2][3] -- see rlenvs
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1
local hist_len = 1
anet:add(nn.SpatialConvolution(hist_len*stateSpec[2][1], 32, 5, 5, 2, 2, 1, 1))
anet:add(nn.ReLU(true))
anet:add(nn.SpatialConvolution(32, 32, 5, 5, 2, 2))
anet:add(nn.ReLU(true))
anet:add(nn.View(-1):setNumInputDims(3))

local convOutputSize = anet:forward(torch.Tensor(1,stateSpec[2][1],stateSpec[2][2],stateSpec[2][3])):size():totable()
local hiddenSize = 128
anet:add(nn.Linear(torch.prod(torch.Tensor(convOutputSize)), hiddenSize))
anet:add(nn.ReLU(true))
-- create critic net sharing parameters with action net 
local cnet = anet:clone('weight','bias') -- clone shared parts from anet at this point
cnet:add(nn.Linear(hiddenSize, 1))
-- finish aciton net
anet:add(nn.Linear(hiddenSize, actionRange))
anet:add(nn.SoftMax())
anet:add(nn.EntropyRegularization(beta))
anet:add(nn.ReinforceCategorical(stochastic))

-- initialize aac
-- set config of aac and optimization method
local config = {tmax = 5, discount = 1}
local optimMethod = optim.nag
local optimConfig = {learningRate = 0.001,
                     momentum = 0.9}
local aac = dprl.aac(anet, cnet, config, optimMethod,optimConfig)

-- initialize async
local loadPackage = function(threadIdx) -- load package on creating agent threads
  require 'rPrint'
  require 'rPrintModule'
  require 'EntropyRegularization'
  require 'xlua'
  -- mask rPrint except thread 1
  if threadIdx ~= 1 then
    rPrint = function () end
    print = function () end
  end
end
-- set config of async
local asyncConfig = {nthread = 4, loadPackage = loadPackage, maxSteps = 10000}

-- set preprocessing functions for state and action
local observationHist 
local preprop = function (observation)
  if not observationHist then
    observationHist = torch.Tensor():typeAs(observation):resize(observation:size()):zero()
  end
  
  local state = observation - observationHist
  observationHist = observation:clone()
  return state
end

local oneHot2Index = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actionPreprop =  function(action)
  local act = action:view(-1)*oneHot2Index
  -- act must be number
  return act
end

-- setup a3c
local a3c = dprl.asyncl(aac, env, asyncConfig, statePreprop, actionPreprop)

local epoch = 200
local stepsPerEpoch = 50000
local testEpisode = 1000
-- set learning report called in each step
local totalReward = 0
local reportT = 0
local learningReport = function(T, trans, agent)
  if __threadid == 1 then
    if T:get() - reportT> 1000 then
      xlua.progress(T:get(), stepsPerEpoch)
      reportT = T:get()
    end
    totalReward = totalReward + trans.r
    if trans.t then
      --print('totalReward at traning', totalReward)
      totalReward = 0
    end
  end
end

--set test report called in each step
local testtotalReward = 0
local averageTotalReward = 0
local testReport = function(s, a, r, ns, t)
  testtotalReward = testtotalReward + r
  if t then
    averageTotalReward = averageTotalReward + testtotalReward
    testtotalReward = 0
  end
end

for i = 1, epoch do
  print('Learning epoch', i)
  a3c:learn(stepsPerEpoch, learningReport)
  print('testing')
  a3c:test(testEpisode,testReport)
  print('Average total reward')
  print(averageTotalReward/testEpisode)
  averageTotalReward = 0
end








