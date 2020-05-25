from marabou.models.tf_idf_models import DumbModel

class model():
    def __init__(self, name, description, image, inp, output):
        self.name = name
        self.description = description
        self.image = image
        self.input = inp
        self.output = output
    
    """
    Factory method return type of model based on the name of the pickle file
    """
    def modelFactory(self):
        if self.name == "dumb_model":
            return DumbModel.deserialize('models/sentiment_analysis/dumb_model')

    def predict(self,inp):
        return self.modelFactory().predict_proba(inp)            