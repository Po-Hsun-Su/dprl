local classic = require 'classic'
local Atari = require 'rlenvs.Atari'
local AtariMod, super = classic.class("AtariMod", Atari)

function AtariMod:_init(opts)
  if opts.poolFrmsSize~= opts.actRep then
    opts.poolFrmsSize = opts.actRep
    print("Warning: change AtariMod option poolFrmSize to",
          opts.actRep)
  end
  if opts.poolFrmsType~='donothing' then
    opts.poolFrmsType = 'donothing' -- sum should be faster than mean       
    print([[Warning: Override AtariMod option poolFrmsType to donothing.
    Please implement pooling in stateProp.]],
          opts.poolFrmsType)
  end
  if opts.randomStarts < opts.poolFrmsSize then
    opts.randomStarts = opts.poolFrmsSize
    print([[Warning: randomStarts must be larger than poolFrmsSize.
    Change AtariMod option randomStarts to]], opts.poolFrmsSize)
  end
  -- put 'donothing into meta table of tensor
  local meta
  if self.gpu and self.gpu >= 0 then
    meta =  getmetatable(torch.CudaTensor)
  else
    meta =  getmetatable(torch.FloatTensor)
  end
  meta['donothing'] = function(arg1, arg2, arg3)
    -- (arg1, arg2, arg3) = (res, src, dim) or (src, dim)
    if arg3 then -- (res, src, dim)
      arg1 = arg2
    end
    return arg1 
  end  
  
  super._init(self, opts)
  self.opts = opts
  self.opts.noOp = self.opts.randomStarts -- ensure last index is at the end of frame buffer
  
end

function AtariMod:getStateSpec()
  return {'real', {self.opts.actRep, 3, 210, 160}, {0, 1}}
end

function AtariMod:start()
  local screen, reward, terminal
  
  if self.gameEnv._random_starts > 0 then
    screen, reward, terminal = self.gameEnv:newGame()
    -- random no op steps
    -- noop must be larger than frame buffer size to fill buffer
    local noop = math.random(self.opts.poolFrmsSize, self.opts.noOp)
    -- skip frame more than poolFrmsSize
    for i = 1, noop - self.opts.poolFrmsSize do
      self.gameEnv.game:play(0)
    end
    -- reset buffer and record poolFrmsSize number of frame
    self.gameEnv._screen.lastIndex = 0 -- reset buffer index, so that first frame is at begin of buffer  
    self.gameEnv._screen.full = false
    for i = 1, self.opts.poolFrmsSize do
      self.gameEnv:_step(0)
    end    
  else
    screen, reward, terminal = self.gameEnv:newGame()
  end
  
  return self.gameEnv._screen.frameBuffer
end

function AtariMod:step(action)
  -- Map action index to action for game
  action = self.actions[action]
  -- Step in the game
  local screen, reward, terminal = self.gameEnv:step(action, self.trainingFlag)
  
  return reward, self.gameEnv._screen.frameBuffer, terminal
end

return AtariMod