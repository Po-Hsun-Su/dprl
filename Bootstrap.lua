--[[

Bootstrap is a container module which contains the bootstrapped heads of a
neural network. See "Deep exploration via bootstrapped DQN" for more detail.

Active head is set through "setActiveHead". This function is also added to nn.module
and nn.Container for broadcasting call.

Output of Bootstrap can be tensor or table denpending on parameter "active".
If "active" is a number index, output will be a tensor.
If "active" is a table containing multiple indexes, output will be a table of tensors.
On testing, output will always be a tensor. The output of multiple heads will be
aggregated. 

The structure of gradOuput is the same as output.    

]]--


local Bootstrap, parent = torch.class('nn.Bootstrap', 'nn.Module')


function Bootstrap:__init(module, headNum, param_init)
    parent.__init(self)
    
    self.headNum = headNum
    self.active = {}
    self.param_init = param_init or 0.1
    self.module = module:clearState()
    self.heads = {}
    self.heads_container = nn.Container()
    self.train = true

    -- initialize heads 
    for k=1,self.headNum do
        if self.param_init then
            -- By default nn.Linear multiplies with math.sqrt(3)
            self.heads[k] = self.module:clone()
            self.heads[k]:reset(self.param_init / math.sqrt(3))
        else    
            self.heads[k] = self.module:clone()
            self.heads[k]:reset()
        end
        self.heads_container:add(self.heads[k])
    end
end

function Bootstrap:clearState()
    self.active = {}
    self.heads_container:clearState()
    return parent.clearState(self)
end

function Bootstrap:parameters(...)
    return self.heads_container:parameters(...)
end

function Bootstrap:type(type, tensorCache)
    return parent.type(self, type, tensorCache)
end
-- add setActiveHead to nn.Container
function nn.Module:setActiveHead(active)
end
function nn.Container:setActiveHead(active)
  self:applyToModules(function(module) module:setActiveHead(active) end)
end

function Bootstrap:setActiveHead(active)
  self.active = active
end

function Bootstrap:updateOutput(input)
    --print(self.active)
    assert( type(self.active) == 'number' or #self.active>0, 'Active head is empty')
    --print('Bootstrap updateOutput')
    --print('input', input)
    
    if type(self.active) == 'number' then
      self.output = self.heads[self.active]:updateOutput(input)
    else
      if self.train then -- Put outputs of heads in a table
        self.output = {}
        for i=1,#self.active do
          self.output[i] = self.heads[self.active[i]]:updateOutput(input)
        end
      else --testing: aggregate outputs of heads
        self.output = self.heads[self.active[1]]:updateOutput(input):clone()
        for i=2,#self.active do
          self.output:add(self.heads[self.active[i]]:updateOutput(input))
        end
        self.output:div(#self.active)
      end
    end
    
    --print('self.output', self.output)
    return self.output
end
-- Only update gradInput of active heads, beacause each head has its own target.
-- bdqn will swith between heads to update all of the heads 
function Bootstrap:updateGradInput(input, gradOutput)
    assert(type(self.active) == 'number' or #self.active>0, 'Active head is empty')
    
    if type(self.active) == 'number' then
      self.gradInput = self.heads[self.active]:updateGradInput(input, gradOutput)
    else
      self.gradInput = self.heads[self.active[1]]:updateGradInput(input, gradOutput[1])
      for i=2,#self.active do
        self.gradInput:add(self.heads[self.active[i]]:updateGradInput(input, gradOutput[i]))
      end
      self.gradInput:div(#self.active)
    end
    
    return self.gradInput
end

function Bootstrap:accGradParameters(input, gradOutput, scale)
    assert(type(self.active) == 'number' or #self.active>0, 'Active head is empty') 
    if type(self.active) == 'number' then
      self.heads[self.active]:accGradParameters(input, gradOutput, scale) 
    else
      -- accumulate grad parameters
      for i=1,#self.active do
          self.heads[self.active[i]]:accGradParameters(input, gradOutput[i], scale)    
      end
    end
end

function Bootstrap:__tostring__()
  local str = 'nn.Bootstrap: '.. self.headNum .. ' heads \n'
  str = str .. '  ' .. self.module:__tostring__()
  
  return str
end