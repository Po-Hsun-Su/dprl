--[[
Double deep Q network
]]--

local classic = require 'classic'
local dqn = require 'dqn'
local ddqn, super = classic.class('ddqn', dqn)

function ddqn:_init(qnet, config, optim, optimConfig)
  super._init(self, qnet, config, optim, optimConfig)
end


function ddqn:setTarget(sampleTrans)
  -- compute the target of each transition
  local mbNextState = torch.Tensor():resize(self.config.batchSize, sampleTrans[1].ns:size(1))
  for i = 1, self.config.batchSize do
    mbNextState[i] =  sampleTrans[i].ns
  end
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
  
  -- Target of each transition 
  for i = 1, self.config.batchSize do
    local sam = sampleTrans[i]
    if sam.t then
      sam.y = sam.r
    else
      -- evaluate value with target action value function
      sam.y = sam.r + self.config.discount*TqValue[i][maxID[i][1]]
    end
  end
  
  return sampleTrans
end

return ddqn