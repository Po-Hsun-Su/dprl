require 'rPrint'
require 'optim'
require 'nn'
require 'rPrintModule'

local qt = pcall(require, 'qt')
local image = require 'image'
local dprl = require 'init'

-- initialize environment
local Catch = require 'rlenvs.Catch'
local env = Catch()
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
 
local stateDim = stateSpec[2][1]*stateSpec[2][2]*stateSpec[2][3] -- see rlenvs
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1
qnet:add(nn.SpatialConvolution(stateSpec[2][1], 32, 5, 5, 2, 2, 1, 1))
qnet:add(nn.ReLU(true))
qnet:add(nn.SpatialConvolution(32, 32, 5, 5, 2, 2))
qnet:add(nn.ReLU(true))
qnet:add(nn.View(-1):setNumInputDims(3))

local convOutputSize = qnet:forward(torch.Tensor(1,stateSpec[2][1],stateSpec[2][2],stateSpec[2][3])):size():totable()
--local hiddenSize = 8
qnet:add(nn.Linear(torch.prod(torch.Tensor(convOutputSize)), actionRange))
--qnet:add(nn.ReLU(true))
--qnet:add(nn.Linear(hiddenSize, actionRange))
local cuda = false
if cuda then
  require 'cunn'
  qnet:cuda()
end
--print(qnet:forward(torch.zeros(1,stateSpec[2][1],stateSpec[2][2],stateSpec[2][3]):cuda()))
-- initialize dqn
local optimConfig = {learningRate = 0.01,
                     momentum = 0.99}
local optimMethod = optim.rmsprop
local dqn_param = {replaySize = 1e5, batchSize = 32, discount = 0.99, epsilon = 0.1}
local dqn = dprl.dqn(qnet,dqn_param, optimMethod, optimConfig)
-- initialize dql
local dql_param = {step = 128, lr = 0.01, updatePeriod = 2000}

local envSize = stateSpec[2][2]
local linspace = torch.linspace(1, envSize, envSize):reshape(envSize,1)
local preprop = function (observation)
                  --print(observation)
                  if cuda then
                    return observation:cuda()
                  else
                    return observation
                  end
                end
local oneHot2ID = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actPreprop = function (action)
                      return action*oneHot2ID
                    end
local report
if qt then
  local window = image.display({image=torch.zeros(1,env.size, env.size), zoom=20})
  report = function(dql_test, totalReward, initQvalue)
    local observation = dql_test.dqn.state:view(1,env.size, env.size)
    image.display({image=observation, zoom=20, win=window})
    print('totalReward',totalReward)   
    print('initQvalue',initQvalue)          
  end
else        
  report = function(dql_test, totalReward, initQvalue)
    print('totalReward',totalReward)   
    print('initQvalue',initQvalue)          
  end
end
local dql = dprl.dql(dqn, env, dql_param, preprop, actPreprop)
print('initial exploration')
optimConfig.learningRate = 0
dqn_param.epsilon = 1
dql:learning(100,report)
print('Learning begain')
optimConfig.learningRate = 0.01
local epsilonDecay = 0.5
for i = 1, 100 do
  dqn_param.epsilon = dqn_param.epsilon*epsilonDecay
  dql:learning(1000,report)
end


print('Press enter to continue')
io.read()

local visualization
if qt then
  local window = image.display({image=torch.zeros(1,env.size, env.size), zoom=20})
  visualization = function (dql, reward)
    local observation = dql.dqn.state:view(1,env.size, env.size)
    if qt then
      image.display({image=observation, zoom=20, win=window})
    end
  end
else
  visualization = function (dql, reward)
    local observation = dql.dqn.state:view(1,env.size, env.size)
    print(observation)
  end
end
dql:test(100, visualization)
print('saving...')
torch.save('test.dql',dql)



