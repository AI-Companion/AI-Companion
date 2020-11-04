Marabou is a python pipeline to perform training and evaluation for different deep learning scenarios. It is distributed under the Mit license.
## Installation
### Depenencies
#### Evaluation mode (default)
- Python (> 3.6)
- ds-gear
- flask (=>1.1.2)
- flask_restful
- pillow
- keras-contrib: Need to be installed separately first (refer to https://github.com/keras-team/keras-contrib)
#### Training mode
- numpy (>=1.19.0rc2)
- pandas (>=1.0.4)
- scikit-learn (>=0.23.1)
- matplotlib
- jupyter
- nltk (>=3.5)
- keras (2.3.1)
- tensorflow (2.2.0)
- opencv-python

### User installation
#### Pre-requisites installation
Make sure to install `keras-contrib` according to https://github.com/keras-team/keras-contrib)  
#### Evaluation
Move to the root of the repository (containing `setup.py`)  
`pip install .`
#### Training
training mode will also install evaluation mode  
Move to the root of the repository (containing `setup.py`)  
`pip install .[train]`

## Source code
You can check the latest sources with the command:
git clone https://github.com/AI-Companion/ds-gear.git