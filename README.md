# Estimating coherent errors with Detector Error Models (DEMs)

This is a c++ package to simulate circuit-level coherent errors for a repetition code memory. 


# Prerequisites

# Installing and compiling
- Create a virtual environment locally. 
```shell
virtualenv coherent_noise
source coherent_noise/bin/activte
cd coherent noise
```

-Install pymatching.
```shell
pip install pymatching
```


- Git clone this repository to your virtual environment.
- Move to the root folder (Estimating_coherent_errors_with_DEMs) 
- Do git submodule init and then git submodule update. This will link the external packages need to run the code.
- Create a build folder and move into the build folder
- Compile via: cmake -DPython_EXECUTABLE=$(which python3) .. and then make -j$(sysctl -n hw.ncpu)


# This is a comment in a shell script example


