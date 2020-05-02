from setuptools import setup, find_packages


INSTALL_REQUIREMENTS = [
    'numpy',
    'pandas',
    'pytest',
    'scikit-learn',
    'matplotlib',
    'flask',
    'flask_restful',
    'pylint'
    ]
setup(name='marabou',
      packages=find_packages(),
      version='0.0.1dev1',
      entry_points={
          'console_scripts': ['marabou-train=marabou.scripts.train_chain:main',
                              'marabou-eval=marabou.scripts.eval_chain:main',
                              'marabou-rest-api=marabou.scripts.serve_rest:main']
      },
      install_requires=INSTALL_REQUIREMENTS,
      package_data={},
      include_package_data=True)