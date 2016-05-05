--[[
bdqn: Bootstrapped deep Q network
The "act" method of bdqn implements deep exploration.   

]]--

local classic = require 'classic'
local ddqn = require 'ddqn'
local bdqn, super = classic.class('bdqn', ddqn)

function bdqn:_init(qnet, headNum, config, optim, optimConfig)
  self.headNum =headNum
  self.allActive = {}
  for i = 1, headNum do self.allActive[i] = i end
  super._init(self, qnet, config, optim, optimConfig)
  
end

function bdqn:act(state, active)
  assert(type(active)=='number', 'the type of active should be number in act')
  self.state = state:clone()
  if state:dim() == 1 then -- add minibatch dimension 
    state = state:view(1,state:size(1))
  end
  --print('state', state)
  if not self.action then -- initialize self.action
    local output = self.qnet:forward(state)
    self.action = torch.Tensor(output:size(2))
  end 
  
  self.action = self.action:zero()
  self.qnet:setActiveHead(active)
  print('state', state)
  print(self.qnet)
  local Qvalue = self.qnet:forward(state)
  self.Qvalue = Qvalue:clone()
  print('Qvalue', Qvalue)
  local maxValue, maxID = torch.max(Qvalue,2)
  --print('maxID',maxID)
  self.action[maxID[1][1]] = 1

  return self.action
end

-- override setTarget
function bdqn:setTarget(sampleTrans)
  local mbNextState = torch.Tensor():resize(self.config.batchSize, sampleTrans[1].ns:size(1))
  for i = 1, self.config.batchSize do
    mbNextState[i] =  sampleTrans[i].ns
    sampleTrans[i].y = {} -- initialize target storage
  end
  self.Tqnet:setActiveHead(self.allActive)
  local qValue = self.Tqnet:forward(mbNextState)
  
  -- target of each transition of each head
  for k = 1, self.headNum do
    local maxQ, maxID = torch.max(qValue[k], 2) -- max Q of each transition 
    -- Target of each transition
    for i = 1, self.config.batchSize do
      local sam = sampleTrans[i]
      if sam.t then
        sam.y[k] = sam.r
      else
        sam.y[k] = sam.r + self.config.discount*maxQ[i][1]
      end
    end
  end
  return sampleTrans
end

-- override learn
-- full sharing
function bdqn:learn(sampleTrans)
  sampleTrans = self:setTarget(sampleTrans)
  for k = 1, self.headNum do
    self.qnet:setActiveHead({k})
    self.Tqnet:setActiveHead({k})
  end
   
end

return bdqn