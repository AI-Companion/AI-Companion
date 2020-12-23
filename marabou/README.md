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
#### Pre-requisites
1. Update pip `pip install --upgrade pip`
2. Setup the environment variable `MARABOU_HOME` to point to the root of this repo `export MARABOU_HOME=/path/to/root`
3. Create a python3 venv and source it (better than mixing with your existing packages):
`python3 -m venv /path/to/your/venv`
`source /path/to/your/venv/bin/activate`
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

## Usage
- evaluation server `marabou-eval-server`
- training `marabou-train-topic-detection`

## Source code
You can check the latest sources with the command:
`git clone https://github.com/AI-Companion/marabou.git`
