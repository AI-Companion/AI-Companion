from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import argparse
from marabou.dumb_model import DumbModel

app = Flask(__name__)
api = Api(app)

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

def main():
    args = parse_arguments()
    model = DumbModel.deserialize(args.model_file)  # load model at the beginning once only
    api.add_resource(PredictSentiment, '/',resource_class_args={model})
    app.run(debug=True)

if __name__ == '__main__':
    main()
