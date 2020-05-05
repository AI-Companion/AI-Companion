from setuptools import setup, find_packages


INSTALL_REQUIREMENTS = [
    'numpy',
    'pandas',
    'pytest',
    'scikit-learn',
    'matplotlib',
    'flask',
    'flask_restful',
    'pylint',
    'doxypypy',
    'pycodestyle'
    #'git_pep8_commit_hook'
    ]
setup(name='marabou',
    #package_dir={'':'marabou'},
    packages=find_packages(include=['.*','marabou','marabou.*']),
    version='0.0.1dev1',
    zip_safe=False,
    entry_points={
        'console_scripts': ['marabou-train=marabou.scripts.train_chain:main',
                            'marabou-eval=marabou.scripts.seval_chain:main',
                            'marabou-rest-api=marabou.scripts.serve_rest:main']
    },
    install_requires=INSTALL_REQUIREMENTS,
    package_data={},
    include_package_data=True)