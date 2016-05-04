--[[
bdql: Bootstrapped deep Q learning

]]--

local classic = require 'classic'
local dql = require 'dql'
local bdql, super = classic.class('bdql', dql)

function bdql:_init(bdqn, env, config, statePreprop, actPreprop)
  super._init(self, bdqn, env, config, statePreprop, actPreprop)
  self.headNum = self.dqn.qnetHead.headNum
end

function bdql:learning(episode, report)
  local updateCounter = 0
  for e = 1, episode do
    -- Sample index of active head
    local active = {math.random(self.headNum)}
    self.active = active
    -- initialize state
    local observation = self.env:start()
    local state = self.statePreprop(observation)
    --print('init state', state)
    for t = 1, self.config.step do 
      local action = self.dqn:act(state, active)
      local actionProp = self.actPreprop(action)
      --print('action', action)
      --print('actionProp', actionProp)
      local reward, observation, terminal = self.env:step(actionProp)
      --print('reward', reward)
      --print('terminal', terminal)
      local nextState = self.statePreprop(observation) -- assume fully observable
      --print('nextState', nextState)
      
      local trans = {s = state:clone(), a = action:clone(), r = reward,
                     ns = nextState:clone(), t = terminal}
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
    if report then report(self) end
  end
  return self.dqn
end

return bdql