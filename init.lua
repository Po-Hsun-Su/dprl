require 'torch'
require 'nn'
require 'dpnn'

require 'classic'

local dprl = {}

dprl.memory = require 'dprl.memory'
dprl.dqn = require 'dprl.dqn'
dprl.ddqn = require 'dprl.ddqn'
dprl.dql = require 'dprl.dql'

require 'dprl.Bootstrap'
dprl.bdq = require 'dprl.bdqn'
dprl.bdql = require 'dprl.bdql'

require 'dprl.EntropyRegularization'

dprl.aac = require 'dprl.aac'
dprl.asyncl = require 'dprl.asyncl'



return dprl
