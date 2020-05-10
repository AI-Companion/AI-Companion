import argparse
import sys
import os
from flask import Flask, render_template, request
from flask_restful import reqparse, Api, Resource
from marabou.models.tf_idf_models import DumbModel

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
    def __init__(self, model):
        self.model = model

    def get(self):
        """
        gets the user's query strings.
        The query could either be a single string or a list of multiple strings
        :return: a dictionary containing probilities prediction as value sorted by each string as key
        """
        # use parser and find the user's query
        args = parser.parse_args()
        query_list = args['query'].strip('][').split(',')
        probs = self.model.predict_proba(query_list)
        return self.model.get_output(probs, query_list)


@app.route('/', methods=['POST', 'GET'])
def index():
    """
    index function
    """
    if request.method == 'POST':
        task_content = request.form['content']
        model = DumbModel.deserialize('models/modelfile.pickle')
        new_prediction = PredictSentiment(model=model)  # new instance of PredictSentiment
        output = new_prediction.model.predict_proba([task_content])
        return render_template('index.html', output=output)
    else:
        return render_template('index.html')


def main():
    """main function"""
    if len(sys.argv) < 2:
        print("no args added")
    else:
        args = parse_arguments()
        model = DumbModel.deserialize(args.model_file)  # load model at the beginning once only
        api.add_resource(PredictSentiment, '/', resource_class_args={model})
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()
