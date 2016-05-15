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

local dqn = classic.class('dqn')

function dqn:_init(qnet, config, optim, optimConfig)
  self.qnet = qnet:clone()
  self.parameters, self.gradParameters = self.qnet:getParameters()
  self.Tqnet = self.qnet:clone()
  self.config = config
  self.optim = optim
  self.optimConfig = optimConfig
  self.memory = {}
  self.memoryLast = 0
  self.criterion = nn.MSECriterion()
  self.criterion.sizeAverage = false
end
function dqn:store(trans)
  -- convert action to ByteTensor for faster indexing while learning 
  if trans.a:type() ~= 'torch.ByteTensor' then
    trans.a = trans.a:byte()
  end
  -- store transition "trans"
  self.memoryLast = self.memoryLast + 1
  --print('insert', self.memoryLast)
  self.memory[self.memoryLast] = trans
  -- remove outdated transition
  --print('remove', self.memoryLast - self.config.replaySize)
  self.memory[self.memoryLast - self.config.replaySize] = nil
end

function dqn:sample()
  local sampleTrans = {}
  for i = 1, self.config.batchSize do
    -- Note #self.memory does not equal the number of transitions in menory
    local randRange = self.config.replaySize
    if self.memoryLast < self.config.replaySize then 
      randRange = self.memoryLast
    end
    local randN = math.random(randRange) - 1 
    --print('randN', randN)
    --print('self.memory')
    --rPrint(self.memory)
    --print('sample ID')
    --print(self.memoryLast - randN)
    sampleTrans[i] = self.memory[self.memoryLast - randN]
  end 
  return sampleTrans
end

function dqn:setTarget(sampleTrans)
  -- compute the target of each transition
  local mbNextState = torch.Tensor():resize(self.config.batchSize, sampleTrans[1].ns:size(1))
  for i = 1, self.config.batchSize do
    mbNextState[i] =  sampleTrans[i].ns
  end
  --print('mbNextState')
  --print(mbNextState)
  -- cascade next states
  
  -- compute Q value through target qnet
  local qValue = self.Tqnet:forward(mbNextState)
  --print('qValue')
  --print(qValue)
  local maxQ, maxID = torch.max(qValue, 2) -- max Q of each transition 
  -- Target of each transition 
  for i = 1, self.config.batchSize do
    local sam = sampleTrans[i]
    if sam.t then
      sam.y = sam.r
    else
      sam.y = sam.r + self.config.discount*maxQ[i][1]
    end
  end
  return sampleTrans
end

function dqn:replay(trans)
  -- store transition
  self:store(trans)
  -- sample from memory
  local sampleTrans = self:sample()
  
  
  return sampleTrans
end

function dqn:learn(sampleTrans)
  -- set target
  sampleTrans = self:setTarget(sampleTrans)
 
  -- organize sample transitions into minibatch input
  local mbState = torch.Tensor(self.config.batchSize, sampleTrans[1].s:size(1))
  local mbTarget = torch.Tensor(self.config.batchSize)-- Target is a number
  for i = 1, self.config.batchSize do
    mbState[i] =  sampleTrans[i].s
    mbTarget[i] = sampleTrans[i].y
  end
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
    local ActQvalue = torch.Tensor(self.config.batchSize)
    for i = 1, self.config.batchSize do
      --print('sampleTrans[i].a', sampleTrans[i].a)
      ActQvalue[i] = Qvalue[i][sampleTrans[i].a]
    end
    local f = self.criterion:forward(ActQvalue, mbTarget)
    
    -- estimate df/dW
    local df_do = self.criterion:backward(ActQvalue, mbTarget)
    
    -- Assign gradient 0 to the Q value of unselected actions
    local gradInput = torch.Tensor(Qvalue:size()):zero()
    for i = 1, self.config.batchSize do
      gradInput[i][sampleTrans[i].a] = df_do[i]
    end 
    self.qnet:backward(mbState, gradInput)
    
    return f, self.gradParameters
  end
  self.optim(feval, self.parameters, self.optimConfig)
  
end

function dqn:update()
  self.Tqnet = self.qnet:clone()
end

function dqn:act(state)
  self.state = state:clone()
  if state:dim() == 1 then -- add minibatch dimension 
    state = state:view(1,state:size(1))
  end
  --print('state', state)
  if not self.action then -- initialize self.action
    local output = self.qnet:forward(state)
    self.action = torch.Tensor(output:size(2))
  end 
  
  local rand = math.random()
  if rand > self.config.epsilon then -- greedy
    self.action = self.action:zero()
    local Qvalue = self.qnet:forward(state)
    self.Qvalue = Qvalue:clone()
    --print('Qvalue', Qvalue)
    local maxValue, maxID = torch.max(Qvalue,2)
    --print('maxID',maxID)
    self.action[maxID[1][1]] = 1
  else -- random action
    self.action = self.action:zero()
    local dim = self.action:size(1)
    local randID = math.random(dim)
    --print('randID',randID)
    self.action[randID] = 1
  end
  return self.action
end

return dqn