import argparse
import sys
import os
from typing import List
from flask import Flask, render_template, request
from flask_restful import reqparse, Api, Resource
from marabou.models.sentiment_analysis.tf_idf_models import DumbModel
from marabou.models.sentiment_analysis.rnn_models import RNNModel, DataPreprocessor
from marabou.utils.config_loader import ConfigReader


app = Flask(__name__)
api = Api(app)
global_model_config = list()

def parse_arguments():
    """
    parse script arguments
    """
    script_parser = argparse.ArgumentParser(description="predict sentiment from a given text")
    script_parser.add_argument('model_file', help='model file', type=str)
    return script_parser.parse_args()


parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):
    """
    utility class for the api_resource method
    """
    def __init__(self, model, pre_processor):
        self.model = model
        self.pre_processor = pre_processor
    
    def getFromService(self, input_list: List[str]):
        """
        gets the user's query strings.
        The query could either be a single string or a list of multiple strings
        :return: a dictionary containing probilities prediction as value sorted by each string as key
        """
        if self.model.model_name == "rnn":
            query_list = DataPreprocessor.preprocess_data(input_list, self.pre_processor)
        probs = self.model.predict_proba(query_list)
        return probs

    def get(self):
        """
        gets the user's query strings.
        The query could either be a single string or a list of multiple strings
        :return: a dictionary containing probilities prediction as value sorted by each string as key
        """
        # use parser and find the user's query
        args = parser.parse_args()
        query_list = args['query'].strip('][').split(',')
        if self.model.model_name == "rnn":
            query_list = DataPreprocessor.preprocess_data(query_list, self.pre_processor)
        probs = self.model.predict_proba(query_list)
        return self.model.get_output(probs, query_list)


@app.route('/', methods=['POST', 'GET'])
def index():
    """
    index function
    """
    return render_template('index.html')

@app.route('/sentimentAnalysis', methods=['POST', 'GET'])
def sentimentAnalysis():
    """
    sentimentAnalysis service function
    """
    if request.method == 'POST':
        task_content = request.form['content']
        new_prediction = PredictSentiment(model= global_model_config[0], pre_processor= global_model_config[1])
        output = new_prediction.getFromService([task_content])
        return render_template('sentiment_analysis.html', output=output)
    else:
        return render_template('sentiment_analysis.html')



def main():
    """main function"""

    """ if boolean is true bring the application up"""
    app_up = len(sys.argv) < 2

    if app_up:
        """
        If no arguments passed to the command, default parameters will be applied ( -c config/config_sentiemnt_analysis.json)
        """
        valid_config = ConfigReader("config/config_sentiment_analysis.json")
    else:
        args = parse_arguments()
        valid_config = ConfigReader(args.config)

    pre_processor = None
    model = None
    if valid_config.eval_model_name == "tfidf":
        model = DumbModel.load_model()
    elif valid_config.eval_model_name == "rnn":
        model, preprocessor_file = RNNModel.load_model()
        pre_processor = DataPreprocessor.load_preprocessor(preprocessor_file)
    else:
        raise ValueError("there is no corresponding model file")
    
    global_model_config.extend([model,pre_processor])
    if app_up:
        # the PredictSentiment methode will be executed in the sentimentAnalysis() method
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        print(PredictSentiment(model, pre_processor))


if __name__ == '__main__':
    main()
