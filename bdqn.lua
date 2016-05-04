--[[
bdqn: Bootstrapped deep Q network
The "act" method of bdqn implements deep exploration.   

]]--

local classic = require 'classic'
local ddqn = require 'ddqn'
local bdqn, super = classic.class('bdqn', ddqn)

function bdqn:_init(qnetShare, qnetHead, config, optim, optimConfig)
  self.qnetShare = qnetShare
  self.qnetHead = qnetHead
  local qnet = nn.Sequential():add(self.qnetShare):add(self.qnetHead)
  print(qnet)
  super._init(self, qnet, config, optim, optimConfig)
  
end

function bdqn:act(state, active)
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
  self.qnetHead:setActiveHead(active)
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
function bdqn:setTarget()
  
end
return bdqn