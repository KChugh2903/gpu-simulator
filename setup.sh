# setup.sh
#!/bin/bash

export ROCKET_SIM_ROOT="$(pwd)"
export ROCKET_SIM_INCLUDE="$ROCKET_SIM_ROOT/include"
export ROCKET_SIM_SRC="$ROCKET_SIM_ROOT/src"
export ROCKET_SIM_LIB="$ROCKET_SIM_ROOT/lib"

# Add to system include path
export CPLUS_INCLUDE_PATH="$ROCKET_SIM_INCLUDE:$ROCKET_SIM_SRC:$CPLUS_INCLUDE_PATH"