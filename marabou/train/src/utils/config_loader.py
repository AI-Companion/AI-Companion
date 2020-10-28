import json

class FashionClassifierConfigReader():
    """
    Loads named entity recognition use case parameters from a .json file
    """
    def __init__(self, file_name):
        self.config = self.read_file(file_name)

    def read_file(self, file_name):
        """
        used to load configuration file
        Args:
            file_name: relative path for the configuration .json file
        """
        with open(file_name, "r") as f:
            return json.load(f)

    @property
    def dataset_name(self):
        """
        name of the dataset to be used
        """
        return self.config["dataset_name"]

    @property
    def validation_split(self):
        """
        ratio of split into validation and training
        """
        return self.config["validation_split"]

    @property
    def experimental_mode(self):
        """
        enable if a quick training is required. will select random 2000 observation
        """
        return self.config["experimental_mode"]

    @property
    def batch_size(self):
        """
        mini-batch size
        """
        return self.config["batch_size"]

    @property
    def pretrained_network_name(self):
        """
        name of the pretrained cnn to use
        """
        return self.config["pretrained_network_name"]

    @property
    def pretrained_network_vgg(self):
        """
        url for the vgg network
        """
        return self.config["pretrained_network_vgg16"]

    @property
    def pretrained_network_lenet(self):
        """
        url for the vgg network
        """
        return self.config["pretrained_network_lenet"]

    @property
    def pre_trained_cnn(self):
        """
        whether to use a pretrained cnn
        """
        return self.config["use_pre_trained_cnn"]

    @property
    def image_height(self):
        """
        image height
        """
        return self.config["image_height"]

    @property
    def image_width(self):
        """
        image width
        """
        return self.config["image_width"]

    @property
    def dataset_url(self):
        """
        gdrive fileid for the specified dataset
        """
        return self.config["dataset_url"]

    @property
    def h5_model_url(self):
        """
        gdrive fileid for the trained model
        """
        return self.config["h5_model_url"]

    @property
    def class_file_url(self):
        """
        gdrive fileid for the class file
        """
        return self.config["class_file_url"]
