--[[
bdql: Bootstrapped deep Q learning

]]--

local classic = require 'classic'
local bdql, super = classic.class('dprl.bdql', 'dprl.dql')

function bdql:_init(bdqn, env, config, statePreprop, actPreprop)
  super._init(self, bdqn, env, config, statePreprop, actPreprop)
  self.headNum = bdqn.config.headNum
end

function bdql:learn(episode, report)
  local updateCounter = 0
  
  for e = 1, episode do
    -- Sample index of active head
    local active = math.random(self.headNum)
    self.active = active
    -- initialize state
    local observation = self.env:start()
    local state = self.statePreprop(observation)
    local totalReward = 0
    local states = {}
    local Qvalues = {}
    states[1] = observation
    --print('init state', state)
    local stepsTaken = self.config.step
    for t = 1, self.config.step do 
      local action = self.dqn:act(state, active)
      local actionProp = self.actPreprop(action)
      Qvalues[states[t]] = self.dqn.Qvalue:clone()
      --print('action', action)
      --print('actionProp', actionProp)
      local reward, observation, terminal = self.env:step(actionProp)
      totalReward = totalReward + reward
      --print('reward', reward)
      --print('terminal', terminal)
      local nextState = self.statePreprop(observation) -- assume fully observable
      states[t+1] = observation
      --print('nextState', nextState)
      
      local trans = {s = state:clone(), a = action:clone(), r = reward,
                     ns = nextState:clone(), t = terminal}
      self.dqn:store(trans)
      -- end of step
      state = nextState -- update state
      if terminal then
        stepsTaken = t
        break
      end
    end
    
    -- learning
    for t = 1, stepsTaken do
      self.dqn:learn(self.dqn:sample())
      
      updateCounter = updateCounter + 1
      if updateCounter%self.config.updatePeriod == 0 then
        self.dqn:update()
      end
    end
    
    if report then report(self, totalReward, active,  states, Qvalues) end
  end
  return self.dqn
end
function bdql:test(episode, report)

end
return bdql