import os
import sys
import json
from typing import List
from flask import Flask, render_template, request
from flask_restful import reqparse, Api, Resource
from PIL import Image
from dsg.sentiment_analysis import SAPreprocessor, SARNN, SAConfigReader
from dsg.named_entity_recognition import NERPreprocessor, NERRNN
from marabou.commons.definitions import SA_CONFIG_FILE, NER_CONFIG_FILE, ROOT_DIR
from marabou.evaluation.utils import load_model


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
    def __init__(self, model: SARNN, pre_processor: SAPreprocessor):
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
        query_list = self.pre_processor.preprocess(input_list)
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
    def __init__(self, model:NERRNN, pre_processor:NERPreprocessor):
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
        questions_list_encoded, questions_list_tokenized, n_tokens, _ = self.pre_processor.preprocess(input_list)
        preds = self.model.predict(questions_list_encoded, self.pre_processor.labels_to_idx, n_tokens)
        display_result = self.model.visualize(questions_list_tokenized, preds)
        return display_result


@app.route('/api/namedEntityRecognition', methods=['POST', 'GET'])
def named_entity_recognition():
    """
    named entity recognition service function
    """
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        task_content = data['content']
        new_prediction = PredictEntities(model=global_model_config[2], pre_processor=global_model_config[3])
        output = new_prediction.get_from_service([task_content])
        return json.dumps(output)
    else:
        return None


# clothing classifier callers
'''
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
'''

@app.route('/', methods=['POST', 'GET'])
def index():
    """
    index function
    """
    return render_template('index.html')


def main():
    """ if boolean is true bring the application up"""
    app_up = len(sys.argv) < 2
    if ROOT_DIR is None:
        raise ValueError("Please make sure to set the environment variable MARABOU_HOME to the root of the directory")
    # load SA models
    valid_config = SAConfigReader(SA_CONFIG_FILE)
    h5_model_file, class_file, preprocessor_file = load_model(h5_file_url=valid_config.h5_model_url,
                                                              class_file_url=valid_config.class_file_url,
                                                              preprocessor_file_url=valid_config.preprocessor_file_url,
                                                              collect_from_gdrive=False,
                                                              use_case="sa")
    if h5_model_file is None or preprocessor_file is None:
        raise ValueError("there is no corresponding SA model file")
    sa_model = SARNN(h5_file=h5_model_file, class_file=class_file)
    sa_preprocessor = SAPreprocessor(preprocessor_file=preprocessor_file)

    # load ner models
    valid_config = SAConfigReader(NER_CONFIG_FILE)
    h5_model_file, class_file, preprocessor_file = load_model(h5_file_url=valid_config.h5_model_url,
                                                              class_file_url=valid_config.class_file_url,
                                                              preprocessor_file_url=valid_config.preprocessor_file_url,
                                                              collect_from_gdrive=False,
                                                              use_case="ner")
    if h5_model_file is None or preprocessor_file is None:
        raise ValueError("there is no corresponding NER model file")
    ner_model = NERRNN(h5_file=h5_model_file, class_file=class_file)
    ner_preprocessor = NERPreprocessor(preprocessor_file=preprocessor_file)

    global_model_config.extend([sa_model, sa_preprocessor, ner_model, ner_preprocessor])
    if app_up:
        # the PredictSentiment methode will be executed in the sentimentAnalysis() method
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, threaded=False)
    else:
        print(PredictSentiment(sa_model, sa_preprocessor))
        print(PredictEntities(ner_model, ner_preprocessor))


if __name__ == '__main__':
    main()
