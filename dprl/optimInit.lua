
assert(optim, 'Load optim package before using optimInit')

local optimInit = function(optimMethod, x, dfdx)
  print('Initializing shared optimization state')
  local state = {}
  if optimMethod == optim.rmsprop then
    state.g = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(1)
  elseif optimMethod == optim.rmspropm then
    state.g = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
    state.gSq = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
  else
    print('WARNING: no initialization scheme found for this optim method')
  end
  return state
end

return optimInit