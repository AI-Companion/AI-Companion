import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


EVALUATION_REQUIREMENTS = [
    'ds-gear',
    'flask==1.1.2',
    'flask_restful',
    'pillow'
]

TRAINING_REQUIREMENTS = [
    'numpy==1.19.0rc2',
    'pandas==1.0.4',
    'scikit-learn==0.23.1',
    'matplotlib',
    'jupyter',
    'nltk==3.5',
    'keras==2.3.1',
    'tensorflow==2.2.0',
    'opencv-python',
    'wget',
    'gdown',
    'requests'
]

class InstallCommand(install):
    """
    will call activate githooks for install mode
    """
    def run(self):
        subprocess.call("git config core.hooksPath .githooks/", shell=True)
        install.run(self)


setup(name='marabou',
      packages=find_packages(include=['commons', 'marabou.*']),
      author='Marouen Azzouz, Youssef Azzouz',
      author_email='azzouz.marouen@gmail.com, youssef.azzouz1512@gmail.com',
      version='0.1.0',
      install_requires=EVALUATION_REQUIREMENTS,
      include_package_data=True,
      zip_safe=False,
      extras_require={
          'train': TRAINING_REQUIREMENTS,
      },
      entry_points={
          'console_scripts': ['marabou-train-sentiment-analysis = marabou.training.scripts.train_sentiment_analysis:main [train]',
                              'marabou-train-named-entity-recognition = marabou.training.scripts.train_named_entity_recognition:main [train]',
                              'marabou-train-fashion-classifier = marabou.training.scripts.train_fashion_classifier:main [train]',
                              'marabou-train-collect-data-fashion-classifier = marabou.training.scripts.cnn_classifier_dataset_collection:main [train]',
                              'marabou-eval-server = marabou.evaluation.app:main'
                              ]
      },
      cmdClass={
          'install': InstallCommand
      })
