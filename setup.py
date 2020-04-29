from setuptools import setup


INSTALL_REQUIREMENTS = [
    'numpy',
    'pandas',
    'pytest',
    'scikit-learn',
    'matplotlib'
]
setup(name='marabou',
      packages=['marabou'],
      version='0.0.1dev1',
      entry_points={
          'console_scripts': ['marabou-train=marabou.scripts.train_chain:main',
                              'marabou-eval=marabou.scripts.eval_chain:main']
      },
      install_requires=INSTALL_REQUIREMENTS
      )