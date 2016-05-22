require 'torch'
require 'nn'
local dprl = {}

dprl.dqn = require 'dqn'
dprl.ddqn = require 'ddqn'
dprl.dql = require 'dql'
require 'Bootstrap'
dprl.bdqn = require 'bdqn'
dprl.bdql = require 'bdql'

return dprl
