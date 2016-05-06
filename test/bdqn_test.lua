require 'rPrint'
local dprl = require 'init'
require 'optim'
require 'nn'

-- Construct qnet

local inputDim = 1
local hiddenDim = 2
local outputDim = 2
local qnetShare = nn.Linear(inputDim,hiddenDim)

local headNum = 2
local param_init = 0.1
local qnetHead = nn.Bootstrap(nn.Linear(hiddenDim,outputDim), headNum, param_init)
local qnet = nn.Sequential():add(qnetShare):add(qnetHead)


local optimConfig = {learningRate = 0.01,
                     momentum = 0.0}
local optimMethod = optim.rmsprop
local dqn_param = {replaySize = 1096, batchSize = 4, discount = 0.99, epsilon = 0.2, headNum = headNum}
local bdqntest = dprl.bdqn(qnet, dqn_param, optimMethod, optimConfig)
print('-----test replay-----')
for i = 1, 10 do
  --print('iter', i)
  local trans = {s = torch.Tensor{i}, a = torch.Tensor{i%2,(i+1)%2}, 
                 r = 0, ns = torch.Tensor{i + 1}, t = false}
  local sample = bdqntest:replay(trans)
  --print('samples')
  --rPrint(sample)
end
io.read()

print('-----test learn-----')
local trans = {s = torch.Tensor{1}, a = torch.Tensor{1,0}, 
                 r = 0, ns = torch.Tensor{2}, t = false}
local sample = bdqntest:replay(trans)
for i = 1, 10 do
  bdqntest:learn(sample)
end
io.read()


print('-----test act-----')
for i = 1, 5 do
  local action = bdqntest:act(torch.Tensor{i},1)
  print('active 1',action)
end

for i = 1, 5 do
  local action = bdqntest:act(torch.Tensor{i},2)
  print('active 2', action)
end
