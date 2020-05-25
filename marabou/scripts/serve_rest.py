import argparse
import sys
import os
from flask import Flask, render_template, request
from flask_restful import reqparse, Api, Resource
from marabou.models.sentiment_analysis.tf_idf_models import DumbModel
from marabou.models.sentiment_analysis.rnn_models import RNNModel, DataPreprocessor
from marabou.utils.config_loader import ConfigReader


app = Flask(__name__)
api = Api(app)


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
    if request.method == 'POST':
        # task_content = request.form['content']
        # model = DumbModel.load_model()
        # new_prediction = PredictSentiment(model=model, pre_processor=pre_processor)
        # output = new_prediction.model.predict_proba([task_content])
        # return render_template('index.html', output=output)
        return None
    else:
        return render_template('index.html')


def main():
    """main function"""
    if len(sys.argv) < 2:
        print("no args added")
    else:
        args = parse_arguments()
        valid_config = ConfigReader(args.config)
        pre_processor = None
        model = None
        if valid_config.eval_model_name == "tfidf":
            model = DumbModel.load_model()
        if valid_config.eval_model_name == "rnn":
            model, preprocessor_file = RNNModel.load_model()
        if model is None:
            raise ValueError("there is no corresponding model file")
        if valid_config.eval_model_name == "rnn":
            pre_processor = DataPreprocessor.load_preprocessor(preprocessor_file)

        api.add_resource(PredictSentiment, '/', resource_class_args={model, pre_processor})
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
