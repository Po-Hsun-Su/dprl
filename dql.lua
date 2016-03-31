--[[
dql: Deep Q learning. It implements the training procedure in [1] using dqn.  
     It is responsible for gluing dqn and env together
Inputs for initialization:
  dqn: deep Q network. See dqn.lua
  env: environment
  param: parameters of the learning process
    step: Max number of steps of each episode
    capacity: capacity of replay memory
    batchsize: batch size of samples from replay memory
    discount: discount in cumulative reward
    
    
  preprop: preprocessing function (optional)
  

methods:
  learning:
    inputs:
      episode: number of episodes  
      report: a function called after each episode (optional)
    output:
      dqn: learned dqn

]]--

local classic = require 'classic'

local dql = classic.class('dql')

function dql:_init(dqn, env, param, statePreprop, actPreprop, report)
  self.dqn = dqn
  self.env = env
  self.param = param
  print('self.param')
  rPrint(self.param)
  self.statePreprop = statePreprop or function(observation) return observation end
  self.actPreprop = actPreprop or function (act) return act end
end

function dql:learning(episode, report)
  local report = report or function(...) return ... end
  local updateCounter = 0
  for e = 1, episode do
    -- initialize state
    local observation = self.env:start()
    local state = self.statePreprop(observation)
    print('init state', state)
    for t = 1, self.param.step do 
      local action = self.dqn:act(state)
      local actionProp = self.actPreprop(action)
      print('action', action)
      print('actionProp', actionProp)
      local reward, observation, terminal = self.env:step(actionProp)
      print('reward', reward)
      print('terminal', terminal)
      local nextState = self.statePreprop(observation) -- assume fully observable
      print('nextState', nextState)
      
      local trans = {s = state:clone(), a = action:clone(), r = reward,
                     ns = nextState:clone(), t = terminal}
      local sampleTrans = self.dqn:replay(trans)
      self.dqn:learn(sampleTrans, self.param.lr)
      
      updateCounter = updateCounter + 1
      if updateCounter%self.param.updateInterval == 0 then
        self.dqn:update()
      end
      -- end of step
      state = nextState -- update state
      if terminal then
        break
      end
    end 
  end
end

return dql