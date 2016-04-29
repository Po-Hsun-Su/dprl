--[[
Double deep Q network
]]--

local classic = require 'classic'
local dqn = require 'dqn'
local ddqn, super = classic.class('ddqn', dqn)

function ddqn:_init(qnet, config, optim, optimConfig)
  super._init(self, qnet, config, optim, optimConfig)
end




return ddqn