require 'rPrint'
local qt = pcall(require, 'qt')
local image = require 'image'
local dprl = require 'init'
local Catch = require 'rlenvs.Catch'
local env = Catch()
require 'nn'
require 'dpnn'
require 'optim'

local visualization
local averageQ
if qt then
  local window = image.display({image=torch.zeros(1,env.size, env.size), zoom=20})
  visualization = function (state,dql)
    local observation = state:view(1,env.size, env.size)
    if qt then
      image.display({image=observation, zoom=20, win=window})
      print('Qvalue',dql.dqn.Qvalue)
      print('Action', dql.dqn.action)
      io.read()
    end
  end
else
  visualization = function (state)
    local observation = state:view(1,env.size, env.size)
    print(observation)
    
  end
end

local report = function(dql_test)
                 print(dql_test.dqn.Qvalue)
               end

local dql = torch.load('test.dql')
dql.dqn.param.epslon = 0.1
--dql:learning(1000, report)
dql:test(100, visualization)

torch.save('test.dql',dql)