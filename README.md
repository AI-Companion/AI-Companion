# Marabou nlp api

Natural Language Processing API. The goal is to enable non machine learning specialists to leverage the advantages of various nlp use cases using an easy interface. The API offers 2 NLP scenarios:  
- Sentiment analysis: Deriving sentiments in sentences (positive, negative, neutral), in articles or in a stream of data such as tweets or product reviews.
- Named entity recognition: Identifies predefined entities in a given text such as Date, Person, Location ... 

## Online interface
For a simple interface in which you type an expression and get a classified output based on the text semantics,
simply go to http://marabou.herokuapp.com/  
1. Inserting the text you want to analyse
2. You receive a probability for a positive review

## Project setup for command line usage
### Pre-requisites
1. python3 and python3 virtualenv  
2. git lfs (if you intend to use trained models directly for inference) (https://git-lfs.github.com/)  
### Setup
We recommand having a virtual environment for the repository. 
On a linux terminal type:  
`$ python3 -m venv /path/to/your/virtual/env`  
`$ source /path/to/your/virtual/env/bin/activate`  
`$ cd /path/to/work/folder`  
`$ git clone https://github.com/mmarouen/marabou.git && cd marabou`  
`$ python setup.py install`  
If you intend to use the trained models:  
From within the repo `$ git lfs pull` (Slow !)    

## Usage for command line tool
Once the setup script installed, you can run the training, evaluation and rest api scripts for both scenarios
On a linux terminal type  
1. `$ marabou-train-sentiment-analysis` to train the sentiment analysis model  
2. `$ marabou-train-ner` to train the named entity recognition model  
Make sure you have enough computing resources on your machine  
Make sure you have at least 1GB space in your disk !  
Alternatively, you could also open and run the scripts `named_entity_recognition.ipynb` and `sentiment_analysis.ipynb` on google colab if your machine does not have the resources.  
These scripts will simply clone the repo and run the training script on the cloud, so no logic is implemented there  
Once the training is finished all you got to do is manually download the 3 generated model files (*.h5 and class.pkl and preprocessor.pkl) files under models/ to your git in the same directory    
3. `$ marabou-valid-sentiment-analysis --space-separated-expressions` to try the model on a list of expressions  
4. `$ marabou-valid-ner --space-separated-expressions` to try the model on a list of expressions  
5. `$ marabou-serve-rest` launches the online app on your machine  

## Model tuning for training
The training script is actually calling the json files under `config/*.json`  
Several training parameters can be adjusted:  
1. `model_name`: You can train either a RNN or a tfidf based Naive bayes learner
2. `embedding_dimension`: In case you choose to train an RNN model, you can select among multiple embedding dimensions  
For more information, you can refer to each parameter's help under `config/config_sentiment_anylsis.json`  

## Activate pre-commit checks
If you wish to have your code automatically checked for quality before every commit, you can activate the githook  
Under the git repo type `git config core.hooksPath .githooks/`