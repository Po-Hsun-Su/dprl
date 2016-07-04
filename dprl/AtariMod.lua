local classic = require 'classic'
local Atari = require 'rlenvs.Atari'
local AtariMod, super = classic.class("AtariMod", Atari)

function AtariMod:_init(opts)
  if opts.poolFrmsSize~= opts.actRep then
    opts.poolFrmsSize = opts.actRep
    print("Warning: change AtariMod option poolFrmSize to",
          opts.actRep)
  end
  if opts.poolFrmsType~='sum' then
    opts.poolFrmsType = 'sum' -- sum should be faster than mean
    print("Warning: change AtariMod option poolFrmsType to",
          opts.poolFrmsType)
  end
  
  super._init(self, opts)
  self.opts = opts
  self.opts.noOp = self.opts.randomStarts*self.opts.actRep - 1 -- ensure last index is at the end of frame buffer
  
  
  
end

function AtariMod:getStateSpec()
  return {'real', {self.opts.actRep, 3, 210, 160}, {0, 1}}
end

function AtariMod:start()
  local screen, reward, terminal
  
  if self.gameEnv._random_starts > 0 then
    
    screen, reward, terminal = self.gameEnv:nextRandomGame(self.opts.noOp)
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