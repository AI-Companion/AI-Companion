import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


EVALUATION_REQUIREMENTS = [
    'ds-gear',
    'flask',
    'flask_restful',
    'pillow',
    'keras-contrib @ https://github.com/keras-team/keras-contrib/tarball/master#egg=package-1.0'
    ]

TRAINING_REQUIREMENTS = [
    'numpy',
    'gdown',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'jupyter',
    'nltk==3.5',
    'keras',
    'tensorflow',
    'opencv-python',
    'wget'
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
                              'marabou-train-topic-detection = marabou.training.scripts.train_topic_classifier:main [train]',
                              'marabou-train-named-entity-recognition = marabou.training.scripts.train_named_entity_recognition:main [train]',
                              'marabou-train-fashion-classifier = marabou.training.scripts.train_fashion_classifier:main [train]',
                              'marabou-eval-server = marabou.evaluation.app:main'
                              ]
      },
      cmdClass={
          'install': InstallCommand
      })
