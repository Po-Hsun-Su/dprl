local dprl = require 'init'
require 'rPrint'
local qnet = nn.Sequential()

local inputDim = 2
local hiddenDim = 2
local outputDim = 2

local qnetShare = nn.Linear(inputDim,hiddenDim)
qnet:add(qnetShare)

local headNum = 2
local param_init = 0.1
local qnetHead = nn.Bootstrap(nn.Linear(hiddenDim,outputDim), headNum, param_init)
qnet:add(qnetHead)

print(qnet)

local input = torch.ones(2)
print('------test: active is number index------') 
for i = 1, headNum do
  local active = i
  qnet:setActiveHead(active)
  local output = qnet:forward(input)
  local gradInput = qnet:backward(input,output)
  print('active head = ',i) 
  print('output value = ', output)
  print('gradInput value = ', gradInput)
end
io.read()

print('------test: active is table containing single index------\n') 
local allactive = {}
for i = 1, headNum do
  local active = {i}
  qnet:setActiveHead(active)
  local output = qnet:forward(input)

  local gradInput = qnet:backward(input,output)
  print('active head = ')
  rPrint(active) 
  print('output value = ')
  rPrint(output)
  print('gradInput value = ', gradInput)
  allactive[i] = i
end
io.read()

print('------test: active is table containing alss indexes------\n') 
qnet:setActiveHead(allactive)
local output = qnet:forward(input)
local gradInput = qnet:backward(input,output)
print('active head = ')
rPrint(allactive) 
print('output value = ')
rPrint(output)
print('gradInput value = ', gradInput)
io.read()

print('------test: active is table containing alss indexes in evaluation------\n')
qnet:evaluate()
qnet:setActiveHead(allactive)
local output = qnet:forward(input)
print('active head = ')
rPrint(allactive) 
print('output value = ')
rPrint(output)

io.read()
