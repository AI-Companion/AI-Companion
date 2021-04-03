Marabou is a python pipeline to perform training and evaluation for different deep learning scenarios. It is distributed under the Mit license.
## Installation
### Depenencies
#### Evaluation mode (default)
- Python (> 3.6)
- flask
- flask_restful
- pillow
- tensorflow
#### Training mode
- numpy
- pandas
- scikit-lear
- matplotlib
- jupyter
- tensorflow
- opencv-python

### User installation
#### Pre-requisites
1. Update pip `pip install --upgrade pip`
2. Create a python3 venv and source it (better than mixing with your existing packages):
`python3 -m venv /path/to/your/venv`
`source /path/to/your/venv/bin/activate`
3. configure githooks path: go to the root of the repository and run `git config core.hooksPath .githooks`
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
- To use the evaluation with the command line, you can curl the server.  
Below is an example to curl the `topics classifier` use case  
`curl http://localhost:5000/api/topicDetection -d '{"content":["example sentence 1", "example sentence 2"] }' -H 'Content-Type: application/json'`
- training `marabou-train-topic-detection`

## Source code
You can check the latest sources with the command:
`git clone https://github.com/AI-Companion/marabou.git`
