-- RMSProp with momentum as found in "Generating Sequences With Recurrent Neural Networks"
function optim.rmspropm(opfunc, x, config, state)
  -- Get state
  local config = config or {}
  local state = state or config
  local lr = config.learningRate or 1e-2
  local momentum = config.momentum or 0.95
  local epsilon = config.epsilon or 0.01

  -- Evaluate f(x) and df/dx
  local fx, dfdx = opfunc(x)
  -- Initialise storage
  if not state.g then -- state.g can be shared and initialized before this function ever called
    state.g = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
    state.gSq = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
  end
  if not state.tmp then
    state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
    state.tmp2 = torch.Tensor():typeAs(x):resizeAs(dfdx)
  end
    
  -- g = αg + (1 - α)df/dx
  torch.mul(state.tmp, state.g, momentum)
  state.g:add(state.tmp, 1 - momentum, dfdx)
  --state.g:mul(momentum):add(1 - momentum, dfdx) -- Calculate momentum
  -- tmp = df/dx . df/dx
  state.tmp:cmul(dfdx, dfdx) 
  -- gSq = αgSq + (1 - α)(df/dx)^2
  torch.mul(state.tmp2, state.gSq, momentum)
  state.gSq:add(state.tmp2, 1 - momentum, state.tmp)
  --state.gSq:mul(momentum):add(1 - momentum, state.tmp) -- Calculate "squared" momentum
  -- tmp = g . g
  state.tmp:cmul(state.g, state.g)
  -- tmp = (-tmp + gSq + ε)^0.5 -- why is there a -tmp
  state.tmp:neg():add(state.gSq):add(epsilon)
  assert(torch.sum(state.tmp:lt(0))==0, 'state.tmp has negative element')
	state.tmp:sqrt()
  -- Update x = x - lr x df/dx / tmp
  x:addcdiv(-lr, dfdx, state.tmp)
  -- Return x*, f(x) before optimisation
  return x, {fx}
end
