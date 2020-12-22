Marabou is a python pipeline to perform training and evaluation for different deep learning scenarios. It is distributed under the Mit license.
## Installation
### Depenencies
#### Evaluation mode (default)
- Python (> 3.6)
- ds-gear
- flask
- flask_restful
- pillow
- keras-contrib: Need to be installed separately first (refer to https://github.com/keras-team/keras-contrib)
#### Training mode
- numpy
- pandas
- scikit-lear
- matplotlib
- jupyter
- nltk (>=3.5)
- keras
- tensorflow
- opencv-python

### User installation
#### Pre-requisites installation
Make sure to install `keras-contrib` according to https://github.com/keras-team/keras-contrib)  
#### Evaluation
Move to the root of the repository (containing `setup.py`)  
`pip install .`
#### Training
training mode will also install evaluation mode  
1. Training on local machine
Move to the root of the repository (containing `setup.py`)  
`pip install .[train]`
2. Training on google colab
simply upload and run to google colab the files `sentiment_analysis.ipynb` or `named_entity_recognition.ipynb`  
The script will generate model files and performance plots which can be downloaded to the models respository  

## Source code
You can check the latest sources with the command:
git clone https://github.com/AI-Companion/ds-gear.git
