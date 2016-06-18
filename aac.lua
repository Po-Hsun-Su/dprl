local classic = require 'classic'
local memory = require 'memory'
local aac = classic.class('aac')
require 'optim'
local optimInit = require 'optimInit'

function aac:_init(anet, cnet, config, optim, optimConfig)
  self.anet = anet:clone()
  self.cnet = cnet:clone()
  self.actorParameters, self.actorGradParameters = self.anet:getParameters()
  self.criticParameters, self.criticGradParameters = self.cnet:getParameters()
  self.config = config
  
  self.optim = optim
  self.optimConfig = optimConfig
  -- initialize optim state to share it between threads
  self.optimState = {}
  self.optimState.actor = optimInit(self.optim, self.actorParameters, self.actorGradParameters)
  self.optimState.critic = optimInit(self.optim, self.criticParameters, self.criticGradParameters)
  
  self.memory = memory(self.config.tmax)
  if self.criticParameters:type() == 'torch.CudaTensor' then
    self.cnetCriterion = nn.MSECriterion():cuda()
  else
    self.cnetCriterion = nn.MSECriterion()
  end
end

function aac:getParameters()
  return {self.actorParameters, self.criticParameters}
end

function aac:getOptimState()
  return self.optimState
end

function aac:zeroGradParameters()
  self.anet:zeroGradParameters()
  self.cnet:zeroGradParameters()
end

function aac:sync(sharedParameters)
  local sharedActorParameters, sharedCriticParameters = unpack(sharedParameters)
  --print('sharedActorParameters',sharedActorParameters)
  --print('sharedCriticParameters', sharedCriticParameters)
  self.actorParameters:copy(sharedActorParameters)
  self.criticParameters:copy(sharedCriticParameters)
end

function aac:act(state)
  -- add minibatch dimension 
  state = state:view(1,unpack(state:size():totable()))
  self.action = self.anet:forward(state)
  return self.action
end

function aac:store(trans)
  -- store transition "trans"
  self.memory:store(trans)
end

function aac:getReturn(nextState, terminal)
  local R = 0
  if not terminal then
    local value = self.cnet:forward(nextState:view(1,unpack(nextState:size():totable())))
    R = value[1]
  end
  return R
end

-- rewrite this function. Write a3c first
function aac:accGradParameters(nextState, terminal)
  assert(not self.memory.full, 'Number of stored transitions should not exceed tmax')
  local tend = self.memory.index
  local mbState = self.memory.storage.s[{{1,tend},{}}]
  local r = self.memory.storage.r[{{1,tend}}]
  
  -- decide initial return (R)
  local lastStateReturn = self:getReturn(nextState, terminal)
  local R = torch.Tensor():typeAs(self.memory.storage.r):resize(tend)
  
  -- compute return of all past states
  local gamma = self.config.discount
  R[tend] = r[tend] + gamma*lastStateReturn
  for t = tend-1,1, -1 do
    R[t] = r[t] + gamma*R[t+1]
  end
  --print('R',R)
  --print('r',r)
  
  -- accumulate gradient in minibatch
  -- actor
  local A = self.anet:forward(mbState)
  self.fa = R
  self.anet:reinforce(R)
  -- the last layer of anet must be a REINFORCE module in dpnn
  -- Can support other critirion for mixing supervised learning
  local dA_do = torch.Tensor():typeAs(A):resizeAs(A):zero() 
  self.anet:backward(mbState, dA_do)
  -- critic
  local V = self.cnet:forward(mbState) -- evaluate value of all states in memory
  self.fc = self.cnetCriterion:forward(V, R)
  local dfc_do = self.cnetCriterion:backward(V, R)
  self.cnet:backward(mbState, dfc_do)
  
  -- reset memory
  self.memory:reset()
  
  return self.actorGradParameters, self.criticGradParameters
end

function aac:update(sharedParameters, T, t, sharedOptimState)
  local sharedActorParameters, sharedCriticParameters = unpack(sharedParameters)
  local actorFeval = function(x)
    return self.fa, self.actorGradParameters
  end
  
  local criticFeval = function(x)
    return self.fc, self.criticGradParameters
  end

  self.optim(actorFeval, sharedActorParameters, self.optimConfig, sharedOptimState.actor)
  self.optim(criticFeval, sharedCriticParameters, self.optimConfig, sharedOptimState.critic)
end

return aac