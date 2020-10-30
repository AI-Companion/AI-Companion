import os
import sys
import json
from typing import List
from flask import Flask, render_template, request
from flask_restful import reqparse, Api, Resource
from PIL import Image
from models.sentiment_analysis_rnn import RNNModel as SARNN
from models.sentiment_analysis_rnn import DataPreprocessor as SAPreprocessor
from models.named_entity_recognition_rnn import RNNModel as NERRNN
from models.named_entity_recognition_rnn import DataPreprocessor as NERPreprocessor
from models.cnn_classifier import CNNClothing


app = Flask(__name__)
api = Api(app)
global_model_config = list()


parser = reqparse.RequestParser()
parser.add_argument('query')


# sentiment analysis callers
class PredictSentiment(Resource):
    """
    utility class for the api_resource method
    """
    def __init__(self, model, pre_processor):
        self.model = model
        self.pre_processor = pre_processor

    def get_from_service(self, input_list: List[str]):
        """
        gets the user's query strings.
        The query could either be a single string or a list of multiple strings
        Args:
            input_list: textual input
        Return:
            dictionary containing probilities prediction as value sorted by each string as key
        """
        if self.model.model_name == "rnn":
            query_list = SAPreprocessor.preprocess_data(input_list, self.pre_processor)
        probs = self.model.predict_proba(query_list)
        return probs


@app.route('/api/sentimentAnalysis', methods=['POST', 'GET'])
def sentiment_analysis():
    """
    sentiment analysis service function
    """
    if request.method == 'POST':
        task_content = request.json['content']
        new_prediction = PredictSentiment(model=global_model_config[0], pre_processor=global_model_config[1])
        output = new_prediction.get_from_service([task_content])
        return json.dumps(output[0] * 100)
    else:
        return None


# Named entity recognition callers
class PredictEntities(Resource):
    """
    utility class for the api_resource method
    """
    def __init__(self, model, pre_processor):
        self.model = model
        self.pre_processor = pre_processor

    def get_from_service(self, input_list: List[str]):
        """
        gets the user's query strings.
        The query could either be a single string or a list of multiple strings
        Args:
            input_list: textual input
        Return:
            dictionary containing probilities prediction as value sorted by each string as key
        """
        questions_list_encoded, questions_list_tokenized, n_tokens =\
            NERPreprocessor.preprocess_data(input_list, self.pre_processor)
        preds = self.model.predict(questions_list_encoded, self.pre_processor["labels_to_idx"], n_tokens)
        display_result = self.model.visualize(questions_list_tokenized, preds)
        return display_result


@app.route('/api/namedEntityRecognition', methods=['POST', 'GET'])
def named_entity_recognition():
    """
    named entity recognition service function
    """
    if request.method == 'POST':
        task_content = request.json['content']
        new_prediction = PredictEntities(model=global_model_config[2], pre_processor=global_model_config[3])
        output = new_prediction.get_from_service([task_content])
        return json.dumps(output)
    else:
        return None


# clothing classifier callers
class ClothingClassifier(Resource):
    """
    utility class for the api_resource method
    """
    def __init__(self, model):
        self.model = model

    def get_from_service(self, image_url: str):
        """
        gets the user's query strings.
        The query could either be a single string or a list of multiple strings
        Args:
            image_url: url containing image to be tested
        Return:
            dictionary containing probilities prediction as value sorted by each string as key
        """
        image = self.model.load_image(image_url)
        image_class = self.model.predict(image)
        return image_class


@app.route('/api/clothingClassifier', methods=['POST', 'GET'])
def clothing_classifier():
    """
    sentiment analysis service function
    """
    if request.method == 'POST':
        image = request.files['image']
        img = Image.open(image.stream)
        img_loc = os.environ.get("MARABOU_HOME") + "marabou/evaluation/clothing_service_images/" + str(image.filename)
        img.save(img_loc)
        new_prediction = ClothingClassifier(model=global_model_config[4])
        img_class = new_prediction.get_from_service(img_loc)
        return json.dumps(img_class)
    else:
        return None


@app.route('/', methods=['POST', 'GET'])
def index():
    """
    index function
    """
    return render_template('index.html')


def main():
    """ if boolean is true bring the application up"""
    app_up = len(sys.argv) < 2
    root_dir = os.environ.get("MARABOU_HOME")
    if root_dir is None:
        raise ValueError("Please make sure to set the environment variable MARABOU_HOME to the root of the directory")
    if not os.path.isdir(os.path.join(root_dir, "marabou/evaluation/trained_models")):
        raise ValueError("You need to have trained models under marabou/evaluation/trained_models to use the evaluation app")
    sentiment_analysis_model = None
    sentiment_analysis_model, preprocessor_file = SARNN.load_model()
    if sentiment_analysis_model is None or preprocessor_file is None:
        raise ValueError("Please make sure to set the correct model path and Project root \
as environment variable MARABOU_HOME")
    sentiment_analysis_pre_processor = SAPreprocessor.load_preprocessor(preprocessor_file)

    ner_model, preprocessor_file = NERRNN.load_model()
    ner_pre_processor = NERPreprocessor.load_preprocessor(preprocessor_file)
    if ner_model is None or ner_pre_processor is None:
        raise ValueError("Please make sure to set the correct model path and Project root \
as environment variable MARABOU_HOME")

    clothing_model = CNNClothing.load_model()
    if clothing_model is None:
        raise ValueError("Please make sure to set the correct model path and Project root \
as environment variable MARABOU_HOME")

    global_model_config.extend([sentiment_analysis_model, sentiment_analysis_pre_processor,
                                ner_model, ner_pre_processor, clothing_model])
    if app_up:
        # the PredictSentiment methode will be executed in the sentimentAnalysis() method
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, threaded=False)
    else:
        print(PredictSentiment(sentiment_analysis_model, sentiment_analysis_pre_processor))
        print(PredictEntities(ner_model, ner_pre_processor))


if __name__ == '__main__':
    main()
