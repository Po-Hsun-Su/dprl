require 'torch'
-- cmd options 
local cmd = torch.CmdLine()
cmd:option('-threads', 4, 'number of async agent thread')
cmd:option('-load', '', 'load saved parameters')
cmd:option('-beta', 0.01, 'Strength of entropy regularization')
cmd:option('-game', 'breakout', 'Atari game to play')
cmd:option('-lr', 7e-4, 'Learning rate')
cmd:option('-epoch', 1000, "Number of training epoch")
cmd:option('-step', 1e5, "Number of steps per training epoch")
cmd:option('-episode', 50, "Number of testing episode")
cmd:option('-visualize', false, "Visualize state")
cmd:option('-optim', 'rmsprop', "Optimization method for gradient descent")


local opts = cmd:parse(arg)
local defaultTensorType = 'torch.FloatTensor'
torch.setdefaulttensortype(defaultTensorType) -- Atari frame is stored in float Tensor

-- load package
require 'optim'
require 'nn'
require 'dpnn'
local dprl = require 'dprl'
require 'image'

-- initialize environment
local AtariMod = require 'dprl.AtariMod'
local AtariConfig = 
  {game = opts.game,
   actRep = 4,
   poolFrmsType = 'donothing',
   poolFrmsSize = 4,
   randomStarts = 10}
local env = AtariMod(AtariConfig)
local stateSpec = env:getStateSpec()
local actionSpec = env:getActionSpec()

-- construct action network and critic network
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1
local beta = opts.beta
local stochastic = false
local anet = nn.Sequential()
-- input dim 4x84x84
anet:add(nn.SpatialConvolution(4,16,8,8,4,4,3,3))
anet:add(nn.ReLU(true)) 
anet:add(nn.SpatialConvolution(16,32,4,4,2,2,1,1))
anet:add(nn.ReLU(true))
anet:add(nn.View(-1):setNumInputDims(3))
local dummyInput = torch.Tensor(1,4,110,84)
local convOutputSize = anet:forward(dummyInput):size():totable()
local hiddenSize = 256
anet:add(nn.Linear(torch.prod(torch.Tensor(convOutputSize)), hiddenSize))
anet:add(nn.ReLU(true))

-- create critic net that shares parameters with action net 
local cnet = anet:clone('weight','bias','gradWeight','gradBias')
cnet:add(nn.Linear(hiddenSize, 1))

-- finish aciton net
anet:add(nn.Linear(hiddenSize, actionRange))
anet:add(nn.SoftMax())
anet:add(nn.EntropyRegularization(beta))
anet:add(nn.ReinforceCategorical(stochastic))

-- initialize aac
local config = {tmax = 5, discount = 0.99}
local optimName = opts.optim
local optimMethod = optim[optimName]
local optimConfig = {learningRate = opts.lr,
                     momentum = 0.99,
                     alpha = 0.99,
                     epsilon = 0.1,
                     share = true}
local aac = dprl.aac(anet, cnet, config, optimMethod,optimConfig)

-- load parameters if available
if opts.load ~='' then
  local parameters = torch.load(opts.load)
  aac:setParameters(parameters)
  print('Load parameter', opts.load)
end

-- initialize async
local loadPackage = function(threadIdx) -- load package on creating agent threads
  require 'xlua'
  require 'image'
  torch.setdefaulttensortype(defaultTensorType) 
  
end
-- Atari emulator cannot be copied by serialization.
-- Load new Atari emulator in each thread
local loadEnv = function()
  local AtariMod = require 'dprl.AtariMod'
  return AtariMod(AtariConfig)
end

local asyncConfig = {nthread = opts.threads,
                     loadPackage = loadPackage, 
                     loadEnv = loadEnv,
                     maxSteps = 1e5}
                     
-- preprocessing functions
local lastframe 
local statePreprop = function(observation)
  local gray = torch.mean(observation,2):squeeze()
  local frames = image.scale(gray,110,84)
  if not lastframe then
    lastframe = frames[{{1},{}}]
  end
  local state = torch.Tensor():typeAs(frames):resizeAs(frames)
  state[{{1},{}}] = torch.cmax(lastframe, frames[{{1},{}}])
  for i = 2, frames:size(1) do
    state[{{i},{}}] = torch.max(frames[{{i-1,i},{}}],1)
  end
  
  return state
end

local oneHot2Index = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actionPreprop =  function(action)
  local act = action:view(-1)*oneHot2Index
  return act
end

local testActionPreprop = function(action)
  local actMax, actID = torch.max(action:view(-1), 1)
  return oneHot2Index[actID[1]]
end

local rewardPreprop = function(reward)
  return math.min(math.max(reward,-1),1)
end 

-- setup a3c
local a3c = dprl.asyncl(aac, env, asyncConfig, statePreprop, actionPreprop, rewardPreprop)
local epoch = opts.epoch
local stepsPerEpoch = opts.step
local testEpisode = opts.episode

-- set learning report called in each step
local totalReward = 0
local reportT = 0
local learningReport = function(trans, t, T, agent)
  --print(agent.cnet.output)
  if __threadid == 1 then
    if T - reportT> 1000 then
      xlua.progress(T, stepsPerEpoch)
      reportT = T
    end
    totalReward = totalReward + trans.r
    if trans.t then
      --print('totalReward at traning', totalReward)
      --print(' ')
      totalReward = 0
    end
  end
end

--set test report called in each step
local testtotalReward = 0
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

--set test report called in each episode in main thread
local averageTotalReward = 0
local maxTotalReward = 0
local minTotalReward = 1e10
local testEpisodicReport = function(report, e)
  averageTotalReward = averageTotalReward + report
  if maxTotalReward < report then maxTotalReward = report end
  if minTotalReward > report then minTotalReward = report end
end

-- set metaData of training   
local bestAverageTotalReward = 0
local lastAverageTotalReward = 0
local date = os.date("%Y%m%d-%H%M%S")
local metaData = {
  averageTotalReward = {}, 
  optimName = optimName,
  optimConfig = optimConfig 
}

-- set test report for visualization
local visStepReport
local visEpisodeReport
local visAverageTotalReward = 0
if opts.visualize then
  require 'qt' 
  image = require 'image'
  local vistotalReward = 0
  local window = image.display({image = torch.zeros(stateSpec[2][2],stateSpec[2][3],stateSpec[2][4]), zoom = 4})
  visStepReport = function(trans, t, E)
    for f = 1, trans.o:size(1) do
      image.display({image = trans.o[{{f}}]:squeeze(), win = window})
    end
    vistotalReward = vistotalReward + trans.r
    print('reward', trans.r)
    local totalRewardCP = vistotalReward
    if trans.t then
      --print('testtotalReward',testtotalReward)
      vistotalReward = 0
    end
    return totalRewardCP
  end
  visEpisodeReport = function(report, e)
    visAverageTotalReward = visAverageTotalReward + report
  end
end
-- learning loop
for i = 1, epoch do
  -- Learning 
  print('Learning epoch', i)
  local status, err = pcall(a3c.learn, a3c, stepsPerEpoch, learningReport)
  if not status then
  
  end
  -- testing
  print('testing')
  a3c:test(testEpisode,testStepReport, testEpisodicReport, testActionPreprop)
  print('Average total reward')
  averageTotalReward = averageTotalReward/testEpisode
  print(averageTotalReward)
  print('Max total reward')
  print(maxTotalReward)
  print('Min total reward')
  print(minTotalReward)
  
  -- save parameters if we get higher total reward
  metaData.averageTotalReward[i] = averageTotalReward
  if  averageTotalReward > bestAverageTotalReward then
    local sharedParameters = aac:getParameters()
    print('Saving parameters')
    torch.save('a3cAtari'.. opts.game .. date .. '.t7',sharedParameters)
    bestAverageTotalReward = averageTotalReward
  end
  torch.save('a3cAtari'.. opts.game .. date .. '_meta.t7',metaData)
  averageTotalReward = 0
  maxTotalReward = 0
  minTotalReward = 1e10
  
  -- visualize if required
  if opts.visualize then
    a3c:visualize(1,visStepReport, visEpisodeReport)
    print('visAverageTotalReward', visAverageTotalReward)
    visAverageTotalReward = 0
  end
end










