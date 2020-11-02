# Marabou machine learning toolkit

Natural Language Processing and computer vision tool. The goal is to enable non machine learning specialists to leverage the advantages of various nlp/ computer vision use cases using an easy interface. We enable low cost integration of machine learning capabilities in the business use cases  
The API offers 3 scenarios:  
- Sentiment analysis: Deriving sentiments in sentences (positive, negative, neutral), in articles or in a stream of data such as tweets or product reviews.
- Named entity recognition: Identifies predefined entities in a given text such as Date, Person, Location ... 
- Clothing classification: Identifies clothes type (coat, pants, jean, necklaces ...) from a given input image  

## Online interface
The application offers 3 use cases each accessibe through an easy to use interface.  
Simply go to http://ai-companion.com/  
### Sentiment analysis
For a simple interface in which you type an expression and get a classified output based on the text semantics,
1. Inserting the text you want to analyse
2. You receive a probability for a positive review
### Named entity recognition
You got a text containing several entites you'd like to classify (geographical, data, persons, ...)  
1. Inserting the text you want to analyse  
2. The text will be tokenized and each word will be assigned a category  
### Fashion classifier
The toolkit support clothing type detection. For that you need to go to "Deep Fashion" classifier. This is a ML program using state of the art deep learning algorithms to detect what kind of clothing is in the image  
1. upload the image you want to detect
2. the program will analyse the image and feedback the clothin category: jean, skirt, shirt ...

## Activate pre-commit checks
If you wish to have your code automatically checked for quality before every commit, you can activate the githook  
Under the git repo type `git config core.hooksPath .githooks/`

## Contact and further questions
Please do not hesitate to send your questions/remarks to: azzouz.marouen@gmail.com  