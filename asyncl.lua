local classic = require 'classic'
require 'classic.torch' 
local async = classic.class('async')
local threads = require 'threads'
require 'posix'
local tds = require 'tds'
threads.Threads.serialization('threads.sharedserialize')

function async:_init(asyncAgent, env, config, statePreprop, actPreprop)
  self.sharedAgent = asyncAgent
  self.env = env
  self.config = config
  self.statePreprop = statePreprop or function(observation) return observation end
  self.actPreprop = actPreprop or function (act) return act end
  self.T = tds.AtomicCounter()
  -- set up thread pool
  self.pool = threads.Threads(
    config.nthread,
    function(threadIdx) -- load package
      require 'pprint'
      require 'dprl'
      require 'tds'
      require 'optim'
      require 'posix'
      rlenvs = require 'rlenvs'
    end,
    self.config.loadPackage
    ,
    function(threadIdx)
      print('starting async thread ', threadIdx)
      threadAgent = torch.deserialize(torch.serialize(self.sharedAgent)) -- clone agent to global variable
      threadEnv = torch.deserialize(torch.serialize(self.env)) -- clone by serialization. Is there a better way to clone env?
      threadStatePreprop = self.statePreprop
      threadActPreprop = self.actPreprop
    end
  )
  self.pool:specific(true)
end

function async:learn(Tmax, report)
  self.sharedAgent:training()
  -- define asynJob of each actor learner thread 
  local sharedParameters = self.sharedAgent:getParameters()
  local sharedOptimState = self.sharedAgent:getOptimState()
  local T = self.T
  if not report then report = function() end end
  T:set(0)
  local function asynJob()
    -- get stuff from global variable
    local agent = threadAgent
    local env = threadEnv
    local statePreprop = threadStatePreprop
    local actPreprop = threadActPreprop
    -- initialization
    local tstart = 0
    local t = 0
    local state, action, nextState, terminal, reward, observation
      
    -- learning loop
    repeat
      agent:sync(sharedParameters,T,t) -- reset gradent and syncronize
      tstart = t
      if not state then state = statePreprop(env:start()) end -- get state from nextstate or start again
      repeat 
        action = agent:act(state) -- pick action
        reward, observation, terminal = env:step(actPreprop(action)) -- get feedback from environment
        nextState = statePreprop(observation)
        agent:store({s = state, a = action,r = reward}) -- store transition
        T:inc()
        t = t + 1
        report({s = state, a = action,r = reward, ns = nextState, t = terminal})
        state = nextState:clone()
      until terminal or t-tstart == agent.config.tmax
      
      agent:accGradParameters(nextState, terminal)
      agent:update(sharedParameters, T, t, sharedOptimState)
 
      -- reset state if at terminal state 
      if terminal then
        state = nil
        terminal = nil
      end
    until  T:get()>Tmax
    
    return __threadid
  end

  -- add jobs to thread pool
  for i = 1, self.config.nthread do
    self.pool:addjob(i, asynJob,
      function(id)
        print('Async thread ' ..  id .. ' finished')
      end
    )
  end 
  -- wait till all threads finish
  self.pool:synchronize()
end

function async:test(episode, report, actPreprop)
  
  local env = self.env
  local agent = self.sharedAgent
  agent:evaluate()
  local statePreprop = self.statePreprop
  actPreprop = actPreprop or self.actPreprop
  -- default report
  if not report then
    local totalreward = 0
    report = function (s, a, r, ns, t)
      totalreward = totalreward + r
      if t then
        print(totalreward)
        totalreward = 0
      end
    end
  end
  -- test begin
  for e = 1, episode do
    local terminal, nextState, reward, action, observation
    local state = statePreprop(env:start())
    local steps = 0
    while not terminal and steps < self.config.maxSteps do
      action = agent:act(state)
      reward, observation, terminal = env:step(actPreprop(action))
      nextState = statePreprop(observation)
      report(state, action, reward, nextState, terminal)
      state = nextState
      steps = steps + 1
    end
  end  
end

return async





