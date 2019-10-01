# Koopman Eigenfunction Extended Dynamic Mode Decomposition (KEEDMD)
Python library for simulating dynamics and Koopman-based learning of dynamical models.

The code in this repository was prepared to implement the methodology described in 

1. C. Folkestad, D. Pastor, I. Mezic, R. Mohr, M. Fonoberova, J. Burdick, "Extended Dynamic Mode Decomposition with Learned Koopman Eigenfunction for Prediction and Control", in *Proc. American Control Conf*, (submitted) 2020 

The simulation framework of this repository is adapted from the [Learning and Control Core Library](https://github.com/learning-and-control/core).

## macOS setup
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
Create IPython kernel
```
python3 -m ipykernel install --user --name .venv --display-name "Virtual Environment"
```

## Running the code
To run the code, run one of the examples in 
```
keedmd/core/examples
```
currently cart pole and inverted pendulum examples are implemented. Run the example scripts as a module with the root folder of repository as the working directory. For example, in a Python 3 environment run
```
python -m keedmd.core.examples.cart_pole.py
```
