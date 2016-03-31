require 'rPrint'
local dprl = require 'init'

require 'nn'
local qnet = nn.Linear(1,2)
local param = {capacity = 4, batchSize = 2, epslon = 0.1, discount = 1, lr = 0.1, mom = 0.99}
local dqntest = dprl.dqn(qnet,param)

for i = 1, 10 do
  --print('iter', i)
  local trans = {s = torch.Tensor{i}, a = torch.Tensor{i%2,(i+1)%2}, r = 0, ns = torch.Tensor{i + 1}}
  local sample = dqntest:replay(trans)
  --print('samples')
  --rPrint(sample)
end

-- test a new dqn
local dqntest = dprl.dqn(qnet, param)
local trans = {s = torch.Tensor{1}, a = torch.Tensor{1,0}, r = 0, ns = torch.Tensor{2}}
dqntest:update()
local sample = dqntest:replay(trans)
for i = 1, 20 do
  dqntest:learn(sample, 0.1)
end

-- test act
for i = 1, 20 do
  local action = dqntest:act(torch.Tensor{i}, 0.1)
  print(action)
end