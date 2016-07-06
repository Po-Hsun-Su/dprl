--[[
"dqn" means "Deep Q-network".

dqn requires the following inputs on construction. 
  qnet: Neural network model
  config:
    replaySize: the size of replay memory
    batchSize: the size of minibatch
    discount: discount factor of reward
    epsilon: the probability of selecting random action
    
dqn implements the following functions.
  replay: store a transition, sample a minibatch of transitions and compute target q value of sampled transitions
    input: 
      1. a transition: a table containg state (s), action (a), reward (r), next state (ns), and terminal (t). 
        Ex: trans = {s, a, r, ns, t} 
        only reward is number
        t = nil if not terminal state 
    output: a sample of minibatch of transitions and the target Q values of each transitions
    
  learn: take sampled minibatch and target Q values and update given qnet work
    input:
      1. minibatch of transitions and target Q values
      2. lr: learning rate
    output: None
        
  update: update the target Q network by current Q network
    input: None
    output: None
    
  act: sample an action according to current state
    input: 
      1. s: current state
    output: action (Onehot representation)
]]--
local classic = require 'classic'
local memory = require 'dprl.memory'
local dqn = classic.class('dprl.dqn')

function dqn:_init(qnet, config, optim, optimConfig)
  self.qnet = qnet:clone()
  self.parameters, self.gradParameters = self.qnet:getParameters()
  self.Tqnet = self.qnet:clone()
  self.Tparameters = self.Tqnet:getParameters()
  self.config = config
  self.optim = optim
  self.optimConfig = optimConfig
  self.memory = memory(self.config.replaySize)
  self.criterion = nn.MSECriterion()
end

function dqn:getParameters()
  return self.parameters
end

function dqn:setParameters(parameters)
  self.parameters:copy(parameters)
  self:update()
end

function dqn:cuda()
  self.qnet:cuda()
  self.parameters, self.gradParameters = self.qnet:getParameters() -- The storage of parameters is changed after cuda()! 
  self.Tqnet:cuda()
  self.memory:cuda()
  self.criterion:cuda()
end

function dqn:store(trans)
  -- convert action to ByteTensor for faster indexing while learning 
  if trans.a:type() ~= 'torch.ByteTensor' then
    trans.a = trans.a:byte()
  end
  -- store transition "trans"
  self.memory:store(trans)
end

function dqn:sample()
  return self.memory:sample(self.config.batchSize)
end

function dqn:setTarget(sampleTrans)
  -- compute the target of each transition

  local mbNextState = sampleTrans.ns

  --print('mbNextState')
  --print(mbNextState)
  -- cascade next states
  
  -- compute Q value through target qnet
  local qValue = self.Tqnet:forward(mbNextState)
  --print('qValue')
  --print(qValue)
  local maxQ, maxID = torch.max(qValue, 2) -- max Q of each transition

  -- Target of each transition
  -- Value in sampleTrans.t is 0 at terminal state. Otherwise, its value is 1.   
  sampleTrans.y = sampleTrans.r + self.config.discount*maxQ:cmul(sampleTrans.t)
  
  return sampleTrans
end

function dqn:replay(trans)
  -- store transition
  self:store(trans)
  -- sample from memory
  return self:sample() 
end

function dqn:learn(sampleTrans)
  -- set target
  sampleTrans = self:setTarget(sampleTrans)
 
  -- organize sample transitions into minibatch input
  local mbState = sampleTrans.s
  local mbTarget = sampleTrans.y

  --print('mbState')
  --print(mbState)
  --print('mbTarget')
  --print(mbTarget)
  
    
  -- Create closure to evaluate f(x) and df/fx
  
  local feval = function(x)
    if x ~= self.parameters then
      self.parameters:copy(x)
    end
    self.gradParameters:zero()
    
    --forward
    local Qvalue = self.qnet:forward(mbState)
    -- select Q value to the selected action
    local ActQvalue = Qvalue[sampleTrans.a]
   
    local f = self.criterion:forward(ActQvalue, mbTarget)
    
    -- estimate df/dW
    local df_do = self.criterion:backward(ActQvalue, mbTarget)
    
    -- Assign gradient 0 to the Q value of unselected actions
    local gradInput = torch.Tensor():typeAs(Qvalue):resize(Qvalue:size()):zero()

    gradInput[sampleTrans.a] = df_do
    
    self.qnet:backward(mbState, gradInput)
    
    return f, self.gradParameters
  end
  self.optim(feval, self.parameters, self.optimConfig)
  
end

function dqn:update()
  self.Tparameters:copy(self.parameters)
end

function dqn:act(state)

  -- add minibatch dimension 
  state = state:view(1,unpack(state:size():totable()))
  
  --print('state', state)
  if not self.action then -- initialize self.action
    local output = self.qnet:forward(state)
    self.action = torch.Tensor(output:size(2))
  end 
  
  local rand = math.random()
  if rand > self.config.epsilon then -- greedy
    self.action = self.action:zero()
    local Qvalue = self.qnet:forward(state)
    --print('Qvalue', Qvalue)
    local maxValue, maxID = torch.max(Qvalue,2)
    --print('maxID',maxID)
    self.action[maxID[1][1]] = 1
  else -- random action
    self.action = self.action:zero()
    local dim = self.action:size(1)
    self.action[math.random(dim)] = 1
  end
  return self.action
end

return dqn