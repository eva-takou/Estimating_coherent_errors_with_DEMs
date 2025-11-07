# Estimating coherent errors with Detector Error Models (DEMs)

This is a c++/python package to simulate, estimate and decode circuit-level coherent errors of a repetition code memory. This code is used to produce the results of the paper: "Estimating and decoding coherent errors of QEC experiments with detector error models" by E. Takou, and K. R. Brown [https://arxiv.org/abs/2510.23797].



# Prerequisites
- Pymatching is required to decode the DEMs.
- The Eigen, pcg, pybind11 packages are also used. 


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

5. Link external packages needed to run these codes with the commands:

```shell
git submodule init
git submodule update
```

6. Make a build folder and compile via the commands:

```shell
mkdir build
cd build
cmake -DPython_EXECUTABLE=$(which python3) ..
make -j$(sysctl -n hw.ncpu)
```

# Authors
Evangelia Takou

# License
This project is licensed under..



