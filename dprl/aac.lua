local classic = require 'classic'
local memory = require 'dprl.memory'
local aac = classic.class('dprl.aac')
require 'optim'
local optimInit = require 'dprl.optimInit'

function aac:_init(anet, cnet, config, optim, optimConfig)
  assert(torch.isTypeOf(anet.modules[#anet.modules],nn.Reinforce), 'The last module of acter network must be a subclass of nn.Reinforce')
  
  self.anet = anet
  self.cnet = cnet
  self.acnet = nn.ConcatTable():add(self.anet):add(self.cnet) -- pack anet and cnet to one model
  -- get parameters once on acnet
  self.parameters, self.gradParameters = self.acnet:getParameters()  
  assert(self.parameters:nElement() == self.gradParameters:nElement(),
   [[Number of elements of parameters and gradParameters doesn't match. 
   You need to share gradParameters if parameters are shared]])
  self.config = config
  self.config.criticGradScale = self.config.criticGradScale or 0.5
  
  self.optim = optim
  self.optimConfig = optimConfig
  -- initialize optim state to share it between threads
  if optimConfig.share then
    self.optimState = optimInit(self.optim, self.parameters, self.gradParameters)
  else
    self.optimState = {}
  end
  self.memory = memory(self.config.tmax)
  self.cnetCriterion = nn.MSECriterion()
end

function aac:training()
  self.acnet:training()
end


function aac:evaluate()
  self.acnet:evaluate()
end


function aac:getParameters()
  return self.parameters
end
function aac:setParameters(parameters)
  self.parameters:copy(parameters)
end
function aac:getOptimState()
  return self.optimState
end

function aac:zeroGradParameters()
  self.gradParameters:zero()
end

function aac:sync(sharedParameters,T,t)
  self.parameters:copy(sharedParameters)
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
  self:zeroGradParameters()
  assert(not self.memory.full, 'Number of stored transitions should not exceed tmax')
  local tend = self.memory.index
  local mbState = self.memory.storage.s[{{1,tend},{}}]
  local r = self.memory.storage.r[{{1,tend}}]
  local mbAction = self.memory.storage.a[{{1,tend}}]
  -- decide initial return (R)
  local lastStateReturn = self:getReturn(nextState, terminal)
  local R = torch.Tensor():typeAs(self.memory.storage.r):resize(tend)
  
  -- compute return of all past states
  local gamma = self.config.discount
  R[tend] = r[tend] + gamma*lastStateReturn
  for t = tend-1,1, -1 do
    R[t] = r[t] + gamma*R[t+1]
  end
  
  -- accumulate gradient in minibatch
  -- critic
  local V = self.cnet:forward(mbState) -- evaluate value of all states in memory
  self.fc = self.cnetCriterion:forward(V, R)
  local dfc_do = self.cnetCriterion:backward(V, R)
  self.cnet:backward(mbState, dfc_do*self.config.criticGradScale) -- see implementation detail in https://github.com/muupan/async-rl/wiki
  
  -- actor
  -- Shouldn't sample action in forward again. The action sampled here must be the same as the one getting the reward.
  -- Need to feed the actions in memory to anet. i.e. A:copy(mbAction:viewAs(A))
  local A = self.anet:forward(mbState)
  A:copy(mbAction:viewAs(A)) -- override newly sampled action with stored action. Need a less hacky way to do this

  self.fa = R-V
  
  self.anet:reinforce(R-V)
  -- the last layer of anet must be a REINFORCE module in dpnn
  -- Can support other critirion for mixing supervised learning
  local dA_do = torch.Tensor():typeAs(A):resizeAs(A):zero() -- dummy gradient. REINFORCE module ignores it.
  self.anet:backward(mbState, dA_do)
  
  -- reset memory
  self.memory:reset()
  
  return self.gradParameters
end

function aac:update(sharedParameters, T, t, sharedOptimState)

  local feval = function(x)
    -- Note: gradient is computed w.r.t the parameters of thread agent. Not the shared parameter x
    -- fevel doesn't give the gradient at x
    return self.fc, self.gradParameters
  end
  self.optim(feval, sharedParameters, self.optimConfig, sharedOptimState)
end

return aac