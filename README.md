# Koopman Eigenfunction Extended Dynamic Mode Decomposition (KEEDMD)
Python library for simulating dynamics and Koopman-based learning of dynamical models.

The code in this repository was prepared to implement the methodologies described in 

1. C. Folkestad, D. Pastor, I. Mezic, R. Mohr, M. Fonoberova, J. Burdick, "Extended Dynamic Mode Decomposition with Learned Koopman Eigenfunction for Prediction and Control", in *Proc. American Control Conf*, (accepted) 2020 

2. C. Folkestad, D. Pastor, J. W. Burdick, "EEpisodic Koopman Learning of Nonlinear Robot Dynamics with Application to Fast Multirotor Landing", in *Proc. International Conference on Robotics and Automation*, (accepted) 2020 

The simulation framework of this repository is adapted from the [Learning and Control Core Library](https://github.com/learning-and-control/core).

## Setup
Set up virtual environment 
```
python3 -m venv .venv
```
Activate virtual environment
```
source .venv/bin/activate
```
Upgrade package installer for Python
```
pip install --upgrade pip
```
Install requirements
```
pip3 install -r requirements.txt
```

## Running the code
To run the code that demonstrates the method described in [1], run one of the examples
```
core/examples/inverted_pendulum.py
core/examples/cart_pole.py
```
Each example:
- Collects some data with a stabilizing controller
- Compute principal eigenvalues
- Fit the diffeomorphism using PyTorch
- Fit a linear model using SkLearn
- Test prediction error
- Test closed loop error

To run the code that demonstrates the method described in [2], run the example
```
core/examples/episodic_1d_landing.py
```

Run the example scripts as a module with the root folder of repository as the working directory. For example, in a Python 3 environment run
```
python -m core.examples.cart_pole.py
```
