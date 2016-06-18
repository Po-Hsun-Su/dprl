
assert(optim, 'Load optim package before using optimInit')

local optimInit = function(optimMethod, x, dfdx)
  local state = {}
  if optimMethod == optim.rmsprop then
    state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(1)
    state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
  else
    print('WARNING: no initialization scheme found for this optim method')
  end
  return state
end

return optimInit 