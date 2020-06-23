import os
import subprocess
from setuptools import setup, find_packages, Command
from setuptools.command.install import install

INSTALL_REQUIREMENTS = [
    'numpy==1.19.0rc2',
    'flask==1.1.2',
    'flask_restful',
    'nltk==3.5',
    'keras==2.3.1',
    'gunicorn',
    'tensorflow==2.2.0',
    'opencv-python'
]

#test2
class InstallCommand(install):
    """
    will call activate githooks for install mode
    """
    def run(self):
        subprocess.call("git config core.hooksPath .githooks/", shell=True)
        install.run(self)

setup(name='marabou',
    packages=find_packages(include=['.*','src','src.*']),
    author='Marouen Azzouz, Youssef Azzouz',
    author_email='azzouz.marouen@gmail.com, youssef.azzouz1512@gmail.com',
    version='0.1.0',
    zip_safe=False,
    entry_points={
        'console_scripts': ['marabou-evaluation=src.app:main']
    },
    dependency_links = ['git+https://www.github.com/keras-team/keras-contrib.git/@master#egg=keras-contrib'],
    install_requires=[INSTALL_REQUIREMENTS,"keras-contrib"],
    package_data = {},
    include_package_data=True,
    cmdClass={
        'install':InstallCommand
    })