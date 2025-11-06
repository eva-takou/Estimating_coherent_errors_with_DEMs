# Estimating coherent errors with Detector Error Models (DEMs)

This is a c++ package to simulate circuit-level coherent errors for a repetition code memory. 


# Prerequisites

# Installing and compiling
1. Create a virtual environment locally. 
```shell
virtualenv coherent_noise
source coherent_noise/bin/activte
cd coherent noise
```

2. Install pymatching.
```shell
pip install pymatching
```

3. Clone this repository to your virtual environment.
```shell
Git clone https://github.com/eva-takou/Estimating_coherent_errors_with_DEMs.git
```

4. Move to the root folder.

```shell
cd Estimating_coherent_errors_with_DEMs
```

5. Link external packages need to run these codes with the commands:

```shell
git submodule init
git submodule update
```

6. Make a build folder and compile via the commands:

```shell
mkdir build
cd build
cmake -DPython_EXECUTABLE=$(which python3) ..
make -j$(sysctl -n hw
```



