import os
import sys
import json
from typing import List
from flask import Flask, render_template, request
from flask_restful import reqparse, Api, Resource
from PIL import Image
import tensorflow as tf
from marabou.evaluation.utils import load_model
from marabou.commons import SA_CONFIG_FILE, MODELS_DIR, NER_CONFIG_FILE, ROOT_DIR,\
                            SAConfigReader, NERConfigReader, TD_CONFIG_FILE, TDConfigReader,\
                            rnn_classification_visualize, custom_standardization


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
    def __init__(self, model):
        self.model = model

@app.route('/api/sentimentAnalysis', methods=['POST', 'GET'])
def sentiment_analysis():
    """
    sentiment analysis service function
    """
    if request.method == 'POST':
        #data = request.get_json(force=True)
        task_content = request.json['content']
        model_wrapper = PredictSentiment(global_model_config[0])
        output = model_wrapper.model.predict(task_content)
        output = [o[0] for o in output]
        return json.dumps([round(o * 100, 2) for o in output])
    else:
        return None

# Named entity recognition callers
class PredictEntities(Resource):
    """
    utility class for the api_resource method
    """
    def __init__(self, model):
        self.model = model

@app.route('/api/namedEntityRecognition', methods=['POST', 'GET'])
def named_entity_recognition():
    """
    named entity recognition service function
    """
    if request.method == 'POST':
        task_content = request.json['content']
        model_wrapper = PredictEntities(global_model_config[1])
        res = model_wrapper.model.predict(task_content)
        output = [r.decode("utf-8").split(" ") for r in list(res)]
        return json.dumps(output)
    else:
        return None

'''
# topic detection callers
class PredictTopic(Resource):
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
        query = self.pre_processor.clean(input_list)
        query = self.pre_processor.preprocess(query)
        probs = self.model.predict(query)
        return probs


@app.route('/api/topicDetection', methods=['POST', 'GET'])
def topic_detection():
    """
    sentiment analysis service function
    """
    if request.method == 'POST':
        task_content = request.json['content']
        new_prediction = PredictTopic(model=global_model_config[4], pre_processor=global_model_config[5])
        output = new_prediction.get_from_service(task_content)
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
    # load SA models
    valid_config = SAConfigReader(SA_CONFIG_FILE)
    sa_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, valid_config.model_name),
                                        custom_objects={'custom_standardization': custom_standardization})

    # load ner models
    valid_config = NERConfigReader(NER_CONFIG_FILE)
    ner_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, valid_config.model_name))
    """
    # load topic detection models
    valid_config = TDConfigReader(TD_CONFIG_FILE)
    h5_model_file, class_file, preprocessor_file = load_model(h5_file_url=valid_config.h5_model_url,
                                                              class_file_url=valid_config.class_file_url,
                                                              preprocessor_file_url=valid_config.preprocessor_file_url,
                                                              collect_from_gdrive=False,
                                                              use_case="td")
    if h5_model_file is None or preprocessor_file is None:
        raise ValueError("there is no corresponding TD model file")
    td_model = RNNMTO(h5_file=h5_model_file, class_file=class_file)
    td_preprocessor = RNNMTOPreprocessor(preprocessor_file=preprocessor_file)
    """
    global_model_config.extend([sa_model, ner_model])
    if app_up:
        # the PredictSentiment methode will be executed in the sentimentAnalysis() method
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, threaded=False)
    else:
        print(PredictSentiment(sa_model))
        print(PredictEntities(ner_model))
        #print(PredictTopic(td_model, td_preprocessor))


if __name__ == '__main__':
    main()