package = "dprl"
version = "scm-1"

source = {
   url = "git://github.com/PoHsunSu/dprl.git",
   tag = "master"
}

description = {
   summary = "Deep reinforcement learning package for torch7",
   detailed = [[
        Deep reinforcement learning package for torch7
   ]],
   homepage = "https://github.com/PoHsunSu/dprl"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
