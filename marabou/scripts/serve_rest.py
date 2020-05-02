import pickle
from flask import Flask, render_template, url_for, request, redirect
from flask_restful import reqparse, abort, Api, Resource
import argparse
from marabou.dumb_model import DumbModel
import sys

app = Flask(__name__)
api = Api(app)

def parse_arguments():
    parser = argparse.ArgumentParser(description="predict sentiment from a given text")
    parser.add_argument('model_file', help='model file', type=str)

    return parser.parse_args()

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def __init__(self, model):
        self.model = model
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        query = args['query']
        probs = self.model.predict_proba([query])

        # create JSON object
        output = {'class0_probs': probs[0][0], 'class1_probs': probs[0][1]}
        return output


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
