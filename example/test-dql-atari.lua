require 'torch'
local cuda = pcall(require, 'cutorch')
-- cmd options 
local cmd = torch.CmdLine()
cmd:option('-cuda', cuda and 1 or 0, 'GPU device ID (0 to disable)')
cmd:option('-load', '', 'load saved parameters')
cmd:option('-epsilon', 1, 'Starting epsilon of epsilon greedy')
local opts = cmd:parse(arg)

-- load package
require 'optim'
require 'nn'
require 'dpnn'
local dprl = require 'dprl'
local image = require 'image'

-- initialize environment
local AtariMod = require 'dprl.AtariMod'
local AtariConfig = 
  {game = 'breakout',
   actRep = 4,
   poolFrmsType = 'sum',
   poolFrmsSize = 4,
   randomStarts = 1}
local env = AtariMod(AtariConfig)
local stateSpec = env:getStateSpec()
local actionSpec = env:getActionSpec()

-- construct Q network
-- Note that the Q network must operate in minibatch mode
local qnet = nn.Sequential() 
  
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1

qnet:add(nn.SpatialConvolution(4,16,8,8,4,4,3,3))
qnet:add(nn.ReLU(true))
qnet:add(nn.SpatialConvolution(16,32,4,4,2,2,1,1))
qnet:add(nn.ReLU(true))
qnet:add(nn.View(-1):setNumInputDims(3))
local dummyInput = torch.Tensor(1,4,110,84)
local convOutputSize = qnet:forward(dummyInput):size():totable()
local hiddenSize = 256
qnet:add(nn.Linear(torch.prod(torch.Tensor(convOutputSize)), hiddenSize))
qnet:add(nn.ReLU(true))
qnet:add(nn.Linear(hiddenSize, actionRange))

-- initialize dqn
require 'dprl.rmspropm' -- load rmsprop with momentum implemented by Kaixhin
local optimMethod = optim.rmspropm
local optimConfig = {learningRate = 0.00025, 
                     momentum = 0.95}
local dqn_param = {replaySize = 2.5e4, batchSize = 16, discount = 0.99, epsilon = 0.1}
local dqn = dprl.ddqn(qnet,dqn_param, optimMethod, optimConfig)

-- load parameters
if opts.load ~= '' then
  local parameters = torch.load(opts.load)
  dqn:setParameters(parameters)
  print('Load parameter', opts.load)
end

-- initialize dql
local dql_param = {step = 1e6, updatePeriod = 2e3}
local envSize = stateSpec[2][2]

local statePreprop = function(observation)
  local gray = torch.mean(observation,2):squeeze()
  local scaled = image.scale(gray,110,84)
  return scaled:double()
end
local oneHot2ID = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actPreprop = function (action)
  local act = action:view(-1)*oneHot2ID
  return act
end

local dql = dprl.dql(dqn, env, dql_param, statePreprop, actPreprop)

-- use cuda
if opts.cuda ~= 0 then
  require 'cutorch'
  require 'cunn'
  dql:cuda()
  -- Need to convert state to CudaTensor
  dql.statePreprop = function (observation)
    local gray = torch.mean(observation,2):squeeze()
    local scaled = image.scale(gray,110,84)
    return scaled
  end
end

-- learning
print('Fill replay memory')
dql:fillMemory()
print('Learning begain')
local epsilonStart = opts.epsilon
dqn_param.epsilon = epsilonStart
local epsilonEnd = 0.01
local epoch = 1000
local epsilonDecay = math.pow(epsilonEnd/epsilonStart, 1/epoch)
local episode = 100
local testEpisode = 100
local report = function(trans, t, e)
  xlua.progress(e, episode)
end

local totalReturn = 0
local testReport = function(trans, t, e)
  xlua.progress(e, testEpisode)
  totalReturn = totalReturn + trans.r
end

local bestAverageReturn= 0
local date = os.date("%Y%m%d-%H%M%S") 
for i = 1, epoch do
  dqn_param.epsilon = dqn_param.epsilon*epsilonDecay
  print('learn with epsilon =', dqn_param.epsilon)
  dql:learn(episode,report)
  print('test')
  dql:test(testEpisode, testReport)
  local averageReturn = totalReturn/testEpisode
  print('Average return ', averageReturn)
  
  if  averageReturn > bestAverageReturn then
    local sharedParameters = dqn:getParameters()
    print('Saving parameters')
    torch.save('ddqnCatch' .. date .. '.t7',sharedParameters)
    bestAverageReturn = averageReturn
  end
  totalReturn = 0
end




