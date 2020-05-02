import pickle
from flask import Flask, render_template, url_for, request, redirect
from flask_restful import reqparse, abort, Api, Resource
import argparse
import logging
from marabou.dumb_model import DumbModel
import sys

app = Flask(__name__)
api = Api(app)
# logging.config.fileConfig('config/logging.conf')
# log = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="predict sentiment from a given text")
    parser.add_argument('model_file', help='model file', type=str)

    return parser.parse_args()

parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def __init__(self, model):
        self.model = model
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        qlist = args['query'].strip('][').split(',')
        probs = self.model.predict_proba(qlist)
        return self.model.get_output(probs, qlist)


@app.route('/',methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        model = DumbModel.deserialize('models/modelfile.pickle')
        new_prediction = PredictSentiment(model=model) # new instance of PredictSentiment
        output = new_prediction.model.predict_proba([task_content])
        return render_template('index.html', output=output)
    else:
        return render_template('index.html')
        



def main():
    if len(sys.argv) < 2:
        print("no args added")
    else:
        args = parse_arguments()
        model = DumbModel.deserialize(args.model_file)  # load model at the beginning once only
        api.add_resource(PredictSentiment, '/',resource_class_args={model})
    app.run(debug=True)

if __name__ == '__main__':
    main()
