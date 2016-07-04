--[[
dql: Deep Q learning. It implements the training procedure in [1] using dqn.  
     It is responsible for gluing dqn and env together
Inputs for initialization:
  dqn: deep Q network. See dqn.lua
  env: environment
  config: parameters of the learning process
    step: Max number of steps of each episode
    
  
    
  statePreprop: a function preprocess observation from environment to state (optional)
    Ex: Conversion From number to Tensor
  actPreprop: a function preprocess action from dqn:act to the form compatible with environment
    Ex: Conversion of onehot representation to integer index representation 

methods:
  learning:
    inputs:
      episode: number of episodes  
      report: a function called after each episode (optional)
    output:
      dqn: learned dqn
  testing:
    inputs:
      visualization: a function visualizes the state of each step

]]--

require 'xlua'
local classic = require 'classic'
require 'classic.torch' -- for saving/loading using torch.save/torch.load

local dql = classic.class('dprl.dql')

function dql:_init(dqn, env, config, statePreprop, actPreprop)
  self.dqn = dqn
  self.env = env
  self.config = config
  self.config.hist_len = 1
  self.statePreprop = statePreprop or function(observation) return observation end
  self.actPreprop = actPreprop or function (act) return act end
end
function dql:cuda()
  self.dqn:cuda()
end
function dql:fillMemory()
  while not self.dqn.memory.full do
    xlua.progress(self.dqn.memory.index, self.dqn.memory.memorySize)
    local observation = self.env:start()
    local state = self.statePreprop(observation)
    for t = 1, self.config.step do
      local action = self.dqn:act(state)
      local actionProp = self.actPreprop(action)
      local reward, observation, terminal = self.env:step(actionProp)
      local nextState = self.statePreprop(observation) 
      assert(state~=nextState, 'State and nextState should not reference the same tensor')
      local trans = {s = state, a = action:clone(), r = reward,
                     ns = nextState, t = terminal and 0 or 1}
      self.dqn:store(trans)
      state = nextState -- update state
      if terminal then
        break
      end            
    end
  end
end
function dql:learn(episode, report)
  local updateCounter = 0
  for e = 1, episode do
    -- initialize state
    local observation = self.env:start()
    local state = self.statePreprop(observation)
    
    for t = 1, self.config.step do
      local action = self.dqn:act(state)
      local actionProp = self.actPreprop(action)
      local reward, observation, terminal = self.env:step(actionProp)
      
      local nextState = self.statePreprop(observation)
      assert(state~=nextState, 'State and nextState should not reference the same tensor. You can clone state in statePreprop')
      local trans = {s = state, a = action:clone(), r = reward,
                     ns = nextState, t = terminal}
      if report then report(trans, t, e) end
      
      trans.t = trans.t and 0 or 1
      local sampleTrans = self.dqn:replay(trans)
      self.dqn:learn(sampleTrans)

      updateCounter = updateCounter + 1
      if updateCounter%self.config.updatePeriod == 0 then
        self.dqn:update()
      end
      -- end of step
      state = nextState -- update state
      if terminal then
        break
      end
    end
  end
  return self.dqn
end

function dql:test(episode, report)
  for e = 1, episode do
    -- initialize state
    local observation = self.env:start()
    local state = self.statePreprop(observation)
    for t = 1, self.config.step do
      local action = self.dqn:act(state)
      --print('action', action)
      local actionProp = self.actPreprop(action)
      --print('actionProp', actionProp)
      local reward, observation, terminal = self.env:step(actionProp)
      local nextState = self.statePreprop(observation) -- assume fully observable
      local trans = {s = state, a = action, r = reward,
                     ns = nextState, t = terminal}
      if report then report(trans, t, e) end
      -- end of step
      state = nextState -- update state
      if terminal then
        break
      end
    end
  end
end

return dql
