--[[
Entropy regularization for policy gradient.
This module should be added after softmax module and before 
ReinforceCategorical module.

This module has no effect in the forward pass.
However, in the backward pass, this module add gradient that increases
the entropy of the probability distribution of actions to the gradInput
from the ReinforceCategorical module.

On construction, a parameter "beta" specifying the strength of regularization
 is required.

]]--

local EntropyRegularization, parent = torch.class('nn.EntropyRegularization', 'nn.Module')

function EntropyRegularization:__init(beta)
  self.beta = beta
end

function EntropyRegularization:updateOutput(input)
  self.output = input
  return self.output
end

function EntropyRegularization:updateGradInput(input, gradOutput)
  local dE_da = torch.log(input  + 0.00000001) + 1
  --print('torch.log(input)', torch.log(input))
  --print('dE_da',dE_da)
  self.gradInput =  gradOutput + self.beta*dE_da
  --print('self.gradInput',self.gradInput)
  return self.gradInput
end