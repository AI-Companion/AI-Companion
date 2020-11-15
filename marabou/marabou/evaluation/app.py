import os
import sys
import json
from typing import List
from flask import Flask, render_template, request
from flask_restful import reqparse, Api, Resource
from PIL import Image
from dsg.RNN_MTM_classifier import RNNMTMPreprocessor, RNNMTM
from dsg.RNN_MTO_classifier import RNNMTO, RNNMTOPreprocessor
from dsg.CNN_classifier import CNNClassifierPreprocessor, CNNClassifier
from marabou.evaluation.utils import load_model
from marabou.commons import SA_CONFIG_FILE, NER_CONFIG_FILE, ROOT_DIR, SAConfigReader, NERConfigReader, rnn_classification_visualize
from marabou.commons import CC_CONFIG_FILE, CCConfigReader


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
    def __init__(self, model: RNNMTO, pre_processor: RNNMTOPreprocessor):
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
    def __init__(self, model:RNNMTM, pre_processor:RNNMTMPreprocessor):
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
        print(questions_list_encoded)
        print(questions_list_tokenized)
        preds = self.model.predict(questions_list_encoded, self.pre_processor.labels_to_idx, n_tokens)
        display_result = rnn_classification_visualize(questions_list_tokenized, preds)
        return display_result


@app.route('/api/namedEntityRecognition', methods=['POST', 'GET'])
def named_entity_recognition():
    """
    named entity recognition service function
    """
    if request.method == 'POST':
        data = request.get_json()
        task_content = data['content']
        new_prediction = PredictEntities(model=global_model_config[2], pre_processor=global_model_config[3])
        output = new_prediction.get_from_service([task_content])
        return json.dumps(output)
    else:
        return None


# clothing classifier callers

class PredictClothing(Resource):
    """
    utility class for the api_resource method
    """
    def __init__(self, model:CNNClassifier, preprocessor: CNNClassifierPreprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def get_from_service(self, image_url_list: str):
        """
        gets the user's query strings.
        The query could either be a single string or a list of multiple strings
        Args:
            image_url_list: url containing image to be tested
        Return:
            dictionary containing probilities prediction as value sorted by each string as key
        """
        images = self.preprocessor.preprocess(image_url_list)
        image_class = self.model.predict(images)
        return image_class

@app.route('/api/clothingClassifier', methods=['POST', 'GET'])
def clothing_classifier():
    """
    sentiment analysis service function
    """
    if request.method == 'POST':
        task_content = request.json['content']
        new_prediction = PredictClothing(model=global_model_config[4], preprocessor=global_model_config[5])
        img_class = new_prediction.get_from_service(task_content)
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
    sa_model = RNNMTO(h5_file=h5_model_file, class_file=class_file)
    sa_preprocessor = RNNMTOPreprocessor(preprocessor_file=preprocessor_file)

    # load ner models
    valid_config = NERConfigReader(NER_CONFIG_FILE)
    h5_model_file, class_file, preprocessor_file = load_model(h5_file_url=valid_config.h5_model_url,
                                                              class_file_url=valid_config.class_file_url,
                                                              preprocessor_file_url=valid_config.preprocessor_file_url,
                                                              collect_from_gdrive=False,
                                                              use_case="ner")
    if h5_model_file is None or preprocessor_file is None:
        raise ValueError("there is no corresponding NER model file")
    ner_model = RNNMTM(h5_file=h5_model_file, class_file=class_file)
    ner_preprocessor = RNNMTMPreprocessor(preprocessor_file=preprocessor_file)

    # load cnn models
    valid_config = CCConfigReader(CC_CONFIG_FILE)
    h5_model_file, _, preprocessor_file = load_model(h5_file_url=valid_config.h5_model_url,
                                                              class_file_url=valid_config.class_file_url,
                                                              preprocessor_file_url=valid_config.preprocessor_file_url,
                                                              collect_from_gdrive=False,
                                                              use_case="CNN")
    if h5_model_file is None or preprocessor_file is None:
        raise ValueError("there is no corresponding CNN model file")
    cnn_preprocessor = CNNClassifierPreprocessor(preprocessor_file=preprocessor_file)
    idx_to_labels = {v:k for k,v in cnn_preprocessor.labels_to_idx.items()}
    cnn_model = CNNClassifier(h5_file=h5_model_file, idx_to_labels=idx_to_labels)

    global_model_config.extend([sa_model, sa_preprocessor, ner_model, ner_preprocessor, cnn_model, cnn_preprocessor])
    if app_up:
        # the PredictSentiment methode will be executed in the sentimentAnalysis() method
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, threaded=False)
    else:
        print(PredictSentiment(sa_model, sa_preprocessor))
        print(PredictEntities(ner_model, ner_preprocessor))
        print(PredictClothing(cnn_model, cnn_preprocessor))

if __name__ == '__main__':
    main()