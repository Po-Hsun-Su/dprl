require 'rPrint'
local dprl = require 'init'
require 'optim'
require 'nn'
local qnet = nn.Linear(1,2)
local optimConfig = {learningRate = 0.01,
                     momentum = 0.0}
local optimMethod = optim.rmsprop
local dqn_param = {replaySize = 1096, batchSize = 4, discount = 0.99, epsilon = 0.2}
local dqntest = dprl.ddqn(qnet,dqn_param, optimMethod, optimConfig)

for i = 1, 10 do
  --print('iter', i)
  local trans = {s = torch.Tensor{i}, a = torch.Tensor{i%2,(i+1)%2}, 
                 r = 0, ns = torch.Tensor{i + 1}, t = false}
  local sample = dqntest:replay(trans)
  --print('samples')
  --rPrint(sample)
end

-- test a new dqn
local dqntest = dprl.dqn(qnet,dqn_param, optimMethod, optimConfig)
local trans = {s = torch.Tensor{1}, a = torch.Tensor{1,0}, 
               r = 0, ns = torch.Tensor{2}, t = false}
dqntest:update()
local sample = dqntest:replay(trans)

for i = 1, 20 do
  dqntest:learn(sample)
end

-- test act
for i = 1, 20 do
  local action = dqntest:act(torch.Tensor{i})
  --print(action)
end