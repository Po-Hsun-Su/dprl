require 'torch'
require 'nn'
require 'dpnn'
local dprl = {}

dprl.dqn = require 'dqn'
dprl.ddqn = require 'ddqn'
dprl.dql = require 'dql'
require 'Bootstrap'
dprl.bdqn = require 'bdqn'
dprl.bdql = require 'bdql'
dprl.aac = require 'aac'
dprl.async = require 'async'
return dprl
