--[[ An implementation of RMSprop

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.alpha'             : smoothing constant
- 'config.epsilon'           : value with which to initialise m
- 'config.weightDecay'       : weight decay
- 'state'                    : a table describing the state of the optimizer;
                               after each call the state is modified
- 'state.g'                  : leaky sum of squares of parameter gradients,
- 'state.tmp'                : and the square root (with epsilon smoothing)

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function optim.rmsprop(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 7e-4
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 0.01
    local wd = config.weightDecay or 0

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- (2) weight decay
    if wd ~= 0 then
      dfdx:add(wd, x)
    end

    -- (3) initialize mean square values and square gradient storage
    if not state.g then
      state.g = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(1)
    end
		if not state.tmp then
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
		end
    -- (4) calculate new (leaky) mean squared values
    torch.mul(state.tmp, state.g, alpha)
    --state.g:mul(alpha)
    state.tmp:addcmul(1.0-alpha, dfdx, dfdx)
    state.g:copy(state.tmp)
    
    -- (5) perform update
    --torch.add(state.tmp, state.g, epsilon)
    if torch.sum(state.tmp:lt(0)) > 0 then
      local index = state.tmp:lt(0)
      print('state.tmp',state.tmp[index])
      print('state.g',state.g[index])
      print('dfdx',dfdx[index])
      assert(false, 'state.tmp has negative element')
    end
    state.tmp:add(epsilon)
    
    
    state.tmp:sqrt()
    --state.tmp:sqrt(state.g):add(epsilon)
    local dfdx_thres = 10000
    if torch.sum(dfdx:gt(dfdx_thres)) > 0 then
      local index = dfdx:gt(dfdx_thres)
      print('dfdx greater than ' .. dfdx_thres ,dfdx[index])
    end
    x:addcdiv(-lr, dfdx, state.tmp)
    
    -- return x*, f(x) before optimization
    return x, {fx}
end
