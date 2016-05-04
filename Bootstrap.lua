--[[

Bootstrap is a container module which contains the bootstrapped heads of a
neural network. See "Deep exploration via bootstrapped DQN" for more detail.

]]--


local Bootstrap, parent = torch.class('nn.Bootstrap', 'nn.Module')


function Bootstrap:__init(module, headNum, param_init)
    parent.__init(self)
    
    self.headNum = headNum
    self.active = {}
    self.param_init = param_init or 0.1
    --print(module)
    self.module = module:clearState()
    self.heads = {}
    self.heads_container = nn.Container()
    
    -- initialize heads 
    for k=1,self.headNum do
        if self.param_init then
            -- By default nn.Linear multiplies with math.sqrt(3)
            self.heads[k] = self.module:clone():reset(self.param_init / math.sqrt(3))
        else    
            self.heads[k] = self.module:clone():reset()
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
    assert(#self.active>0, 'Active head is empty')
    print('Bootstrap updateOutput')
    print('input', input)
    -- resize output    
    if input:dim() == 1 then
        self.output:resize(self.module.weight:size(1))
    elseif input:dim() == 2 then
        local nframe = input:size(1)
        self.output:resize(nframe, self.module.weight:size(1))
    end
    self.output:zero()
    print('#self.active',#self.active)
    -- select active heads
    for i=1,#self.active do
        print('self.heads[self.active[i]]', self.heads[self.active[i]])
        self.output:add(self.heads[self.active[i]]:updateOutput(input))
    end
    self.output:div(#self.active)
    print('self.output', self.output)
    return self.output
end
-- Only update gradInput of active heads, beacause each head has its own target.
-- bdqn will swith between heads to update all of the heads 
function Bootstrap:updateGradInput(input, gradOutput)
    -- rescale gradients
    gradOutput:div(#self.active)
    
    -- resize gradinput
    self.gradInput:resizeAs(input):zero()

    -- accumulate gradinputs
    for i=1,#self.active do
        self.gradInput:add(self.heads[self.active[i]]:updateGradInput(input, gradOutput))
    end

    return self.gradInput
end

function Bootstrap:accGradParameters(input, gradOutput, scale)
    -- rescale gradients
    gradOutput:div(#self.active)

    -- accumulate grad parameters
    for i=1,#self.active do
        self.heads[self.active[i]]:accGradParameters(input, gradOutput, scale)    
    end
end

function Bootstrap:__tostring__()
  local str = 'nn.Bootstrap: '.. self.headNum .. ' heads \n'
  str = str .. '  ' .. self.module:__tostring__()
  
  return str
end