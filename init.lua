require 'torch'
require 'nn'
require 'dpnn'
local dprl = {}

require 'dprl.memory'
require 'dprl.dqn'
require 'dprl.ddqn'
require 'dprl.dql'

require 'nn.Bootstrap'
require 'dprl.bdqn'
require 'dprl.bdql'

require 'nn.EntropyRegularization'
require 'dprl.aac'
require 'dprl.asyncl'

--[[
dprl.dqn = require 'dqn'
dprl.ddqn = require 'ddqn'
dprl.dql = require 'dql'
require 'Bootstrap'
dprl.bdqn = require 'bdqn'
dprl.bdql = require 'bdql'
dprl.aac = require 'aac'
dprl.asyncl = require 'asyncl'
]]--
return dprl
