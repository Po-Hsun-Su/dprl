require 'rPrint'
require 'optim'
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
--local stateDim = stateSpec[2][2] + 2 -- handcraft feature see "preprop" below
  -- Add hidden layers
local hiddenSize1 = 16
local hiddenSize2 = 16
local hiddenSize3 = 16
local transfer = 'ReLU'
qnet:add(nn.Linear(stateDim, hiddenSize1))
qnet:add(nn[transfer]())
--print(qnet:forward(torch.rand(4,stateDim))) -- test
qnet:add(nn.Linear(hiddenSize1, hiddenSize2))
qnet:add(nn[transfer]())
qnet:add(nn.Linear(hiddenSize2, hiddenSize3))
qnet:add(nn[transfer]())
  -- Output Q values
local actionRange = actionSpec[3][2] - actionSpec[3][1] + 1
qnet:add(nn.Linear(hiddenSize2,actionRange))
--print(qnet:forward(torch.rand(4,stateDim))) -- test

-- initialize dqn
local optimConfig = {learningRate = 0.01,
                     momentum = 0.0}
local optimMethod = optim.rmsprop
local dqn_param = {replaySize = 5000, batchSize = 16, discount = 0.99, epsilon = 0.1}
local dqn = dprl.ddqn(qnet,dqn_param, optimMethod, optimConfig)
-- initialize dql
local dql_param = {step = 128, lr = 0.01, updatePeriod = 2000}

local envSize = stateSpec[2][2]
local linspace = torch.linspace(1, envSize, envSize):reshape(envSize,1)
local preprop = function (observation)
                  --observation = observation:view(envSize,envSize)
                  --print(observation)
                  --local ballx = torch.sum(observation[{{1,envSize-1},{}}]*linspace)
                  --local bally = torch.sum(linspace[{{1,envSize-1},{}}]:t()*observation[{{1,envSize-1},{}}])
                  --local feature = torch.Tensor(envSize+2)
                  --feature[1], feature[2] = ballx, bally
                  --feature[{{3,envSize+2}}] = observation[{{envSize},{}}]
                  --print(ballx, bally)
                  --print(feature)
                  --io.read()
                  return observation:view(-1)
                  --return feature
                end
local oneHot2ID = torch.linspace(actionSpec[3][1], actionSpec[3][2], actionRange)
local actPreprop = function (action)
                      return action*oneHot2ID
                    end                    
local report = function(dql_test)
                 print(dql_test.dqn.Qvalue)
               end
local dql = dprl.dql(dqn, env, dql_param, preprop, actPreprop)
dql:learning(100000,report)

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



