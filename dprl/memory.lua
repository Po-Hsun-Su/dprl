local classic = require 'classic'
local memory = classic.class('dprl.memory')

function memory:_init(memorySize)
  self.memorySize = memorySize
  self.index = 0
  self.full = false
  self.storage = nil
end
function memory:cuda()
  self.usecuda = true
end
function memory:store(data)
  -- initialize self.storage if it's nil
  if not self.storage then
    if type(data) == 'table' then
      self.storage = {}
      self.istable = true
      for key, value in pairs(data) do
        if type(value) == 'number' then
          if self.usecuda then
            self.storage[key] = torch.CudaTensor(self.memorySize)
          else
            self.storage[key] = torch.Tensor(self.memorySize) -- 1D Tensor
          end
        else 
          self.storage[key] = torch.Tensor():typeAs(value):resize(self.memorySize,unpack(value:size():totable()))
        end
      end
    else
      self.storage = torch.Tensor():typeAs(data):resize(self.memorySize,unpack(data:size():totable()))
    end
  end
  -- get index for storing data
  self.index = self.index + 1
  if self.index > self.memorySize then
    self.index = 1
    self.full = true -- this flag tells the method "sample" the range of sampling
  end
  -- store data (Note that data is coppied because tensor to tensor assigment copy data.)
  if self.istable then
    for key, value in pairs(data) do
      self.storage[key][self.index] = value
    end
  else
    self.storage[self.index] = data
  end
end

function memory:sample(size)
  -- range of index
  local range
  if self.full then
    range = self.memorySize
  else
    range = self.index
  end
  -- get sample indices
  local indices = torch.LongTensor(size)
  for i = 1, size do
    indices[i] = math.random(range)
  end
  -- index samples (Note tenor:index() copies data)
  local sample
  if self.istable then
    sample = {}
    for key, value in pairs(self.storage) do
      sample[key] = value:index(1,indices), indices
    end
  else
    sample = self.storage:index(1,indices), indices
  end
  return sample
end

function memory:index(...)
  return self.storage:index(...)
end

function memory:reset() -- reset will not clear storage
  self.index = 0
  self.full = false
end

return memory