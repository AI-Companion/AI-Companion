import subprocess
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from distutils.command.clean import clean


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

#test2
class InstallCommand(install):
    """will call activate githooks for install mode"""
    def run(self):
        subprocess.call("git config core.hooksPath .githooks/",shell=True)
        install.run(self)

setup(name='marabou',
    #package_dir={'':'marabou'},
    packages=find_packages(include=['.*','marabou','marabou.*']),
    author='Marouen Azzouz, Youssef Azzouz',
    author_email='azzouz.marouen@gmail.com, youssef.azzouz1512@gmail.com',
    version='0.0.1dev1',
    zip_safe=False,
    entry_points={
        'console_scripts': ['marabou-train=marabou.scripts.train_chain:main',
                            'marabou-eval=marabou.scripts.seval_chain:main',
                            'marabou-rest-api=marabou.scripts.serve_rest:main']
    },
    install_requires=INSTALL_REQUIREMENTS,
    tests_require=["pytest", ],
    package_data={},
    include_package_data=True,
    cmdClass={
        'install':InstallCommand
    })