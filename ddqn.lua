--[[
Double deep Q network
]]--

local classic = require 'classic'
local ddqn, super = classic.class('dprl.ddqn', 'dprl.dqn')

function ddqn:_init(qnet, config, optim, optimConfig)
  super._init(self, qnet, config, optim, optimConfig)
end


function ddqn:setTarget(sampleTrans)
  -- compute the target of each transition
  local mbNextState = sampleTrans.ns

  --print('mbNextState')
  --print(mbNextState)
  -- cascade next states
  
  -- compute Q value through target qnet
  local TqValue = self.Tqnet:forward(mbNextState)
  local qValue = self.qnet:forward(mbNextState)
  --print('qValue')
  --print(qValue)
  --print('TqValue')
  --print(TqValue)
  -- select action based on current action value function
  local maxQ, maxID = torch.max(qValue, 2) -- max Q of each transition 
  --print('maxID')
  --print(maxID)
  local TmaxQ = TqValue:gather(2,maxID)
  sampleTrans.y = sampleTrans.r + self.config.discount*TmaxQ:cmul(sampleTrans.t)
  
  return sampleTrans
end

return ddqn