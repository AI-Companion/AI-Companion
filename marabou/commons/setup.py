import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


class InstallCommand(install):
    """
    will call activate githooks for install mode
    """
    def run(self):
        subprocess.call("git config core.hooksPath .githooks/", shell=True)
        install.run(self)


setup(name='commons',
      packages=find_packages(),
      author='Marouen Azzouz, Youssef Azzouz',
      author_email='azzouz.marouen@gmail.com, youssef.azzouz1512@gmail.com',
      version='0.1.0',
      zip_safe=False,
      include_package_data=True,
      cmdClass={
          'install': InstallCommand
      })
