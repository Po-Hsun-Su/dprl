--[[

"dqn" means "Deep Q-network"
dqn implements the following functions.
  replay: store a transition and sample a minibatch of transitions given target qnet
    input: 
      1. a transition: a table containg state (s), action (a), reward (r), next state (ns), and terminal (t). 
        Ex: trans = {s, a, r, ns, t} 
        only reward is number
        t = nil if not terminal state
      2. target qnet: 
    output: a sample of minibatch of transitions and the target Q values of each transitions
    
  learn: take sampled minibatch and target Q values and update given qnet work
    input:
      1. minibatch of transitions and target Q values
      2. learning qnet
      3. lr: learning rate
    output: learned qnet
        
  update: update the target Q network by current Q network
    input: current Q network
    output: new target Q network
  act: sample an action according to current state
    input: 
      1. s: current state
      2. qnet: Q network
      3. epslon: probability of random action
    output: action
      
dqn requires the following inputs on construction. 
  qnet:
  param:
    capacity: the capacity of replay memory
    batchSize: the size of minibatch
    discount: discount factor of reward
 
]]--
local classic = require 'classic'

local dqn = classic.class('dqn')

function dqn:_init(qnet, param)
  self.qnet = qnet:clone()
  self.Tqnet = self.qnet:clone()
  self.param = param
  self.memory = {}
  self.memoryLast = 0
  self.criterion = nn.MSECriterion()
end

function dqn:replay(trans)
  -- store transition "trans"
  self.memoryLast = self.memoryLast + 1
  --print('insert', self.memoryLast)
  self.memory[self.memoryLast] = trans
  -- remove outdated transition
  --print('remove', self.memoryLast - self.param.capacity)
  self.memory[self.memoryLast - self.param.capacity] = nil
  
  -- sample from memory
  local sampleTrans = {}
  for i = 1, self.param.batchSize do
    -- Note #self.memory does not equal the number of transitions in menory
    local randRange = self.param.capacity
    if self.memoryLast < self.param.capacity then 
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
  
  -- compute the target of each transition
  local mbNextState = torch.Tensor():resize(self.param.batchSize, sampleTrans[1].ns:size(1))
  for i = 1, self.param.batchSize do
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
  for i = 1, self.param.batchSize do
    local sam = sampleTrans[i]
    if sam.t then
      sam.y = sam.r
    else
      sam.y = sam.r + self.param.discount*maxQ[i][1]
    end
  end
  
  return sampleTrans
end

function dqn:learn(sampleTrans, lr)
  -- organize sample transitions into minibatch input
  local mbState = torch.Tensor(self.param.batchSize, sampleTrans[1].s:size(1))
  local mbTarget = torch.Tensor(self.param.batchSize)-- Target is a number
  for i = 1, self.param.batchSize do
    mbState[i] =  sampleTrans[i].s
    mbTarget[i] = sampleTrans[i].y
  end
  --print('mbState')
  --print(mbState)
  --print('mbTarget')
  --print(mbTarget)
  -- Forward
  local Qvalue = self.qnet:forward(mbState)
  -- select Q value to the selected action
  local ActQvalue = torch.Tensor(self.param.batchSize)
  for i = 1, self.param.batchSize do
    --print('sampleTrans[i].a:byte()', sampleTrans[i].a:byte())
    ActQvalue[i] = Qvalue[i][sampleTrans[i].a:byte()]
  end
  --print('Qvalue',Qvalue)
  --print('ActQvalue',ActQvalue)
  
  local criterionOutput = self.criterion:forward(ActQvalue, mbTarget)
  --print('criterionOutput')
  --print(criterionOutput)
  
  -- Backward
  local criterionBackward = self.criterion:backward(ActQvalue, mbTarget)
  --print('criterionBackward')
  --print(criterionBackward)
  
  -- Assign gradient 0 to the Q value of unselected actions
  local gradInput = torch.Tensor(Qvalue:size()):zero()
  for i = 1, self.param.batchSize do
    gradInput[i][sampleTrans[i].a:byte()] = criterionBackward[i]
  end 
  --print('gradInput')
  --print(gradInput)
  self.qnet:zeroGradParameters()
  self.qnet:backward(mbState, gradInput)
  self.qnet:updateParameters(lr)
end

function dqn:update()
  self.Tqnet = self.qnet:clone()
end

function dqn:act(state)
  if state:dim() == 1 then -- add minibatch dimension 
    state = state:view(1,state:size(1))
  end
  self.state = state
  print('state', state)
  
  if not self.action then -- initialize self.action
    local output = self.qnet:forward(state)
    self.action = torch.Tensor(output:size(2))
  end 
  
  local rand = math.random()
  if rand > self.param.epslon then -- greedy
    self.action = self.action:zero()
    local Qvalue = self.qnet:forward(state)
    print('Qvalue', Qvalue)
    local maxValue, maxID = torch.max(Qvalue,2)
    print('maxID',maxID)
    self.action[maxID[1][1]] = 1
  else -- random action
    self.action = self.action:zero()
    local dim = self.action:size(1)
    local randID = math.random(dim)
    print('randID',randID)
    self.action[randID] = 1
  end
  return self.action
end

return dqn