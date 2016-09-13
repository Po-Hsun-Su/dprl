--[[ 
Visualization of atari agent
]]--
require 'torch'
qt = require 'qt'
gd = require 'gd'
require 'dprl'
local cmd = torch.CmdLine()
cmd:option('-agent','', "Path to saved agent")
cmd:option('-load', '', 'Path saved parameters')
cmd:option('-game', 'breakout', 'Name of Atari game to play')
cmd:option('-episode', 100, "Number of testing episode")
cmd:option('-gif_file', '', "GIF path to write atari screen")

local opts = cmd:parse(arg)

local a3c = torch.load(opts.agent)