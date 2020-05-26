# Marabou nlp api

Natural Language Processing API. The goal is to enable non machine learning specialists to leverage the advantages of various nlp use cases using an easy interface. The API offers several topics such as:  
- Sentiment analysis: Deriving sentiments in sentences (positive, negative, neutral), in articles or in a stream of data such as tweets or product reviews.
- Topic extraction (under development): Assigning tags/categories to the given text according to its content
- Named entity recognition (under development): Identifies predefined entities in a given text such as Date, Person, Location ... 

## Online interface
For a simple interface in which you type an expression and get a classified output based on the text semantics,
simply go to http://marabou.herokuapp.com/  
1. Inserting the text you want to analyse
2. You receive a probability for a positive review

## Project setup for command line usage
We recommand having a virtual environment for the repository. 
On a linux terminal type:  
`$ python3 -m venv /path/to/your/virtual/env`  
`$ source /path/to/your/virtual/env/bin/activate`  
`$ cd /path/to/work/folder`  
`$ git clone https://github.com/mmarouen/marabou.git && cd marabou`  
`$ python setup.py install`  

## Usage for command line tool
Once the setup script installed, you can run the training, evaluation and rest api scripts  
On a linux terminal type  
1. `$ marabou-train-sentiment-analysis` to train the sentiment analysis model  
ps: Make sure you have at least 1GB space in your disk !  
2. `$ marabou-valid-sentiment-analysis --list-of-comma-separated-expressions` to try the model on a list of expressions  
3. `$ marabou-serve-rest` launches the online app on your machine  

## Model tuning for training
The training script is actually calling the json file `config/config_sentiment_anylsis.json`  
Several training parameters can be adjusted:  
1. `model_name`: You can train either a RNN or a tfidf based Naive bayes learner
2. `embedding_dimension`: In case you choose to train an RNN model, you can select among multiple embedding dimensions  
For more information, you can refer to each parameter's help under `config/config_sentiment_anylsis.json`  

## Model evaluation
Once you finished training at least one mode, you can call it up to perform inference agains a given string list.  
The script will fetch the latest trained `eval_model_name` algorithm  
