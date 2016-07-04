local classic = require 'classic'
require 'classic.torch'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
require 'posix'
local tds = require 'tds'
require 'xlua'
local asyncl = classic.class('dprl.asyncl')

function asyncl:_init(asynclAgent, env, config, statePreprop, actPreprop)
  self.sharedAgent = asynclAgent
  self.env = env
  self.config = config
  self.statePreprop = statePreprop or function(observation) return observation end
  self.actPreprop = actPreprop or function (act) return act end
  self.T = tds.AtomicCounter()
  -- set up thread pool
  local loadEnv =  self.config.loadEnv or function ()
    -- clone by serialization. Please provide envLoader if env cannot be serialized like Atari emulator.
    return torch.deserialize(torch.serialize(env)) 
  end
  self.pool = threads.Threads(
    config.nthread,
    function(threadIdx)
      require 'dprl'
      require 'tds'
      require 'optim'
      require 'posix'
    end,
    self.config.loadPackage,
    function(threadIdx)
      -- don't use "self." here. Otherwise, "self" will be serialized  
      print('starting asyncl thread ', threadIdx)
      threadAgent = torch.deserialize(torch.serialize(asynclAgent)) -- clone agent to global variable
      threadEnv = loadEnv()
      threadStatePreprop = statePreprop
      threadActPreprop = actPreprop
    end
  )
  self.pool:specific(true)
end

function asyncl:learn(Tmax, stepReport)
  self.sharedAgent:training()
  -- define asynJob of each actor learner thread 
  local sharedParameters = self.sharedAgent:getParameters()
  local sharedOptimState = self.sharedAgent:getOptimState()
  local T = self.T
  if not stepReport then stepReport = function(trans, t, T) end end
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
        stepReport({s = state, a = action,r = reward, ns = nextState, t = terminal},t, T:get())
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
        --print('asyncl thread ' ..  id .. ' finished')
      end
    )
  end 
  -- wait till all threads finish
  self.pool:synchronize()
end

function asyncl:test(episode, stepReport, episodicReport, actPreprop)
  -- sync agent
  local sharedParameters = self.sharedAgent:getParameters()
  for i = 1, self.config.nthread do
    self.pool:addjob(i, function ()
        threadAgent:sync(sharedParameters,0,0)
      end
    )
  end
  
  -- get alternative actionPreprop function
  actPreprop = actPreprop or self.actPreprop
  
  -- set report
  if not stepReport then
    local totalreward = 0
    stepReport = function (s, a, r, ns, t)
      totalreward = totalreward + r
      local totalrewardCP = totalreward
      if t then
        totalreward = 0
      end
      return totalrewardCP
    end
  end  
  episodicReport = episodicReport or function(reportResult, e) end
  
  -- begin test
  local E = tds.AtomicCounter()
  local maxStep = self.config.maxSteps
  E:set(0)
  local function asynJob()
    local env = threadEnv
    local agent = threadAgent
    local statePreprop = threadStatePreprop
    local reportResult
    E:inc()
    local terminal, nextState, reward, action, observation
    local state = statePreprop(env:start())
    local t = 0
    while not terminal and t < maxStep do
      action = agent:act(state)
      reward, observation, terminal = env:step(actPreprop(action))
      nextState = statePreprop(observation)
      reportResult = stepReport({s = state, a = action,r = reward, ns = nextState, t = terminal},t, E:get())
      state = nextState
      t = t + 1
    end
    return reportResult, E:get()
  end
  self.pool:specific(false)
  for e = 1, episode do
    self.pool:addjob(asynJob, episodicReport)
  end
  self.pool:synchronize()
  self.pool:specific(true)
  --[[
  for e = 1, episode do
    xlua.progress(e,episode)
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
  ]]--
end

return asyncl





