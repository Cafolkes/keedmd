# KEEDMD
This repository 

Forked from a python simulation and hardware library for learning and control
## Usage
Execute the examples in 'core/examples' as modules from the root folder:
```
python3 -m core.examples.cart_pole
```
or
```
python3 -m core.examples.inverted_pendulum
```
Each example:
- Collects some data with a stabilizing controller
- Compute principal eigenvalues
- Fit the diffeomorphism using PyTorch
- Fit a linear model using SkLearn
- Test prediction error
- Test closed loop error

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
Create IPython kernel
```
python3 -m ipykernel install --user --name .venv --display-name "Virtual Environment"
```
