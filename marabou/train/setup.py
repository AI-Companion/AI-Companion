import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


INSTALL_REQUIREMENTS = [
    'numpy==1.19.0rc2',
    'pandas==1.0.4',
    'pytest',
    'scikit-learn==0.23.1',
    'matplotlib',
    'flask==1.1.2',
    'flask_restful',
    'pylint',
    'doxypypy',
    'pycodestyle',
    'nltk==3.5',
    'keras==2.3.1',
    'tensorflow==2.2.1',
    'opencv-python'
]


class InstallCommand(install):
    """
    will call activate githooks for install mode
    """
    def run(self):
        subprocess.call("git config core.hooksPath .githooks/", shell=True)
        install.run(self)


setup(name='marabou',
      packages=find_packages(include=['marabou', 'marabou.*']),
      author='Marouen Azzouz, Youssef Azzouz',
      author_email='azzouz.marouen@gmail.com, youssef.azzouz1512@gmail.com',
      version='0.1.0',
      zip_safe=False,
      entry_points={
          'console_scripts': ['marabou-train-sentiment-analysis=src.scripts.train_sentiment_analysis:main',
                              'marabou-eval-sentiment-analysis=src.scripts.eval_sentiment_analysis:main',
                              'marabou-train-ner=src.scripts.train_named_entity_recognition:main',
                              'marabou-eval-ner=src.scripts.eval_named_entity_recognition:main',
                              'marabou-train-fashion-classifier=src.scripts.train_fashion_classifier:main',
                              'marabou-eval-fashion-classifier=src.scripts.eval_fashion_classifier:main']
      },
      dependency_links=['git+https://www.github.com/keras-team/keras-contrib.git#egg=keras-contrib'],
      install_requires=INSTALL_REQUIREMENTS,
      tests_require=["pytest", ],
      package_data={},
      include_package_data=True,
      cmdClass={
          'install': InstallCommand
      })
