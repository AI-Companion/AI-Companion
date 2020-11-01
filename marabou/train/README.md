# Marabou training app

The training app is used to prototype, train and evaluate your models for all 3 different use cases.  
Models can be trained either on your local machine using the `train_*.py` files or on google colab using the wrapper scripts `train_*.ipynb`. Unless you have a powerful machine with GPU, we strongly recommend training the models on google colab.  
For that you just need to login to colab and upload the notebook, then launch.  

## Project setup for command line usage
### Pre-requisites
1. python3 and python3 virtualenv  
2. ubuntu 18 (not tested on other distributions)  
### Setup
We recommand having a virtual environment for the repository. 
On a linux terminal type:  
`$ python3 -m venv /path/to/your/virtual/env`  
`$ source /path/to/your/virtual/env/bin/activate`  
`$ cd /path/to/work/folder`  
`$ git clone https://github.com/mmarouen/marabou.git && cd marabou`  
`$ python setup.py install`  

## Training your own models
Once the setup script installed, you can run the training, evaluation and rest api scripts for both scenarios
On a linux terminal type  
1. `$ marabou-train-sentiment-analysis` to train the sentiment analysis model  
2. `$ marabou-train-ner` to train the named entity recognition model  
3. `$ marabou-train-fashion-classifier` to train the clothing classification model  
Make sure you have enough computing resources on your machine  
Make sure you have at least 1GB space in your disk !  
Alternatively, you could also train your models on google colab, to do so
1. open google colab (https://colab.research.google.com/notebooks/intro.ipynb#recent=true)  
2. select the corresponding jupyter notebook provided in the repo `named_entity_recognition.ipynb` or `sentiment_analysis.ipynb` or `fashion_classifier.ipynb`  
3. run the file cell by cell  
These scripts will clone the repo and run the training script on the cloud, so no logic is implemented there  
Once the training is finished all you got to do is manually download the 3 generated model files (*.h5 and class.pkl and preprocessor.pkl) files under models/ to your git in the same directory   

## Evaluate the trained moels
1. `$ marabou-valid-sentiment-analysis --space-separated-expressions` to try the model on a list of expressions  
2. `$ marabou-valid-ner --space-separated-expressions` to try the model on a list of expressions  
3. `$ marabou-train-fashion-classifier --path-to-image` to evaluate the class of the clothing

## Model tuning
The training script is actually calling the json files under `config/*.json`  
Several training parameters can be adjusted:  
1. `model_name`: You can train either a RNN or a tfidf based Naive bayes learner  
2. `embedding_dimension`: In case you choose to train an RNN model, you can select among multiple embedding dimensions  
For more information, you can refer to each parameter's help under `config/config_sentiment_anylsis.json`  
