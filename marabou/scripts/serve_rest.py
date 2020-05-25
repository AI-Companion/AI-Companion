import argparse
import sys
import os
import json
from flask import Flask, render_template, request
from flask_restful import reqparse, Api, Resource
from marabou.models.modelConfig import model

app = Flask(__name__)
api = Api(app)
models_list = []

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
    index function parse availble models under models/sentiment_analysis from models.json file
    """
    models_list = []
    try:
        with open('models/sentiment_analysis/models.json','r') as f:
            conf = json.load(f)
        # load instances of models
        for m in conf:
            models_list.append(model(m['name'], m['description'], m['image'], m['input'], m['output']))
        return render_template('index.html',models = models_list)
    except IOError:    
        print("models.json file not found")
        return render_template('index.html')


@app.route('/model/<model_name>', methods=['POST', 'GET'])
def model_page(model_name):
    try:
        with open('models/sentiment_analysis/models.json','r') as f:
            conf = json.load(f)
        # load instance of chosen model
        for m in conf:
            if m['name'] == model_name:
                model_chosen = model(m['name'], m['description'], m['image'], m['input'], m['output'])

    except IOError:    
        print("models.json file not found")

    if request.method == 'POST':
        try:
            content = request.form['content']
            output = model_chosen.predict([content])
            return render_template('model.html', model = model_chosen, output = output)
        except:
            return render_template('model.html', model = model_chosen)

    else:
        return render_template('model.html', model = model_chosen)


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
