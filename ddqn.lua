--[[
Double deep Q network
]]--

local classic = require 'classic'
local dqn = require 'dqn'
local ddqn, super = classic.class('ddqn', dqn)

function ddqn:_init(qnet, config, optim, optimConfig)
  super._init(self, qnet, config, optim, optimConfig)
end

-- override method replay
function ddqn:replay(trans)
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
  
  -- sample from memory
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