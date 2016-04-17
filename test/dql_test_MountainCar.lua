require 'rPrint'
require 'optim'
require 'math'
local qt = pcall(require, 'qt')
local image = require 'image'
local dprl = require 'init'

-- initialize environment
local MountainCar = require 'rlenvs.MountainCar'
local env = MountainCar()

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
 
local stateDim = 2 -- position and velocity

  -- Add hidden layers
local hiddenSize = 64

local transfer = 'ReLU'
qnet:add(nn.Linear(stateDim, hiddenSize))
qnet:add(nn[transfer]())
--print(qnet:forward(torch.rand(4,stateDim))) -- test

  -- Output Q values
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1
qnet:add(nn.Linear(hiddenSize,actionRange))
--print(qnet:forward(torch.rand(4,stateDim))) -- test

-- initialize dqn
local optimConfig = {learningRate = 0.01,
                     momentum = 0.0}
local optimMethod = optim.rmsprop

local dqn_param = {replaySize = 256, batchSize = 16, discount = 0.99, epslon = 0.1}
local dqn = dprl.dqn(qnet, dqn_param, optimMethod, optimConfig)
-- initialize dql
local dql_param = {step = 256, lr = 0.01, updatePeriod = 64}

local preprop = function (observation)
                  return torch.Tensor(observation)
                end
local oneHot2ID = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actPreprop = function (action)
                      return action*oneHot2ID
                    end  
local report = function(dql_test)
                 print(dql_test.dqn.Qvalue)
               end
local dql = dprl.dql(dqn, env, dql_param, preprop, actPreprop)
dql:learning(1000,report)

local visualization
if qt then
  local width = 200
  local height = 100
  local window = image.display({image=torch.zeros(1, height, width), zoom=5})
  visualization = function(dql, reward)
    local x = (dql.dqn.state[2] - stateSpec[2][3][1])/(stateSpec[2][3][2] - stateSpec[2][3][1])
    local y = (1 + math.sin(3*dql.dqn.state[2]))/2
    local visual = torch.zeros(1,height, width)
    print('x',x)
    print('math.ceil(x*width)', math.ceil(x*width))
    visual[1][math.ceil((1-y)*(height-1))+1][math.ceil(x*(width-1)) + 1] = 1
    image.display({image=visual, zoom=5, win=window})
    print('state')
    print(dql.dqn.state)
    print('action')
    print(dql.dqn.action)
    print('reward')
    print(reward)
    --io.read()
  end
else
  visualization = function(dql, reward)
    print('state')
    print(dql.dqn.state)
    print('action')
    print(dql.dqn.action)
    print('reward')
    print(reward)
    io.read()
  end
end
dql:test(100,visualization)


