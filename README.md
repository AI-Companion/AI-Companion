# Marabou nlp api

Natural Language Processing API. The goal is to enable anyone to leverage the advantages of natrual language processing using a user friendly interface. The API offers several topics such as:
- Sentiment analysis: Deriving sentiments in sentences (positive, negative, neutral), in articles or in a stream of data such as tweets or product reviews.
- Topic extraction (also known as text classification): Assigning tags/categories to the given text according to its content
- Named entity recognition: Identifies predefined entities in a given text such as Date, Person, Location ... 
Usage is a simple 2 steps process:
1. Choosing the nlp topic you want to try
2. Inserting the text you want to analyse
The service is accessible on http://marabou.herokuapp.com/

## Setup
If you want to clone the repository and install it on your machine, we recommand having a virtual environment for that purpose. On your terminal type:  
`$ python3 -m venv /path/to/your/virtual/env`  
`$ source /path/to/your/virtual/env/bin/activate`  
Then clone this repo and move to the root  
`$ python setup.py install`  

## Usage
The application contains 3 runnables:
1. marabou-train: to train the models
2. marabou-valid: to test the model performance
3. marabou-rest-api: to start a web-based rest api service