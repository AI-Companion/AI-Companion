import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


INSTALL_REQUIREMENTS = [
    'bs4',
    'stem',
    'requests',
    'requests[socks]',
    'fake_useragent'
]


class InstallCommand(install):
    """
    will call activate githooks for install mode
    """
    def run(self):
        install.run(self)


setup(name='bastet',
      packages=find_packages(),
      author='Marouen Azzouz, Youssef Azzouz',
      author_email='azzouz.marouen@gmail.com, youssef.azzouz1512@gmail.com',
      version='0.1.0',
      zip_safe=False,
      classifiers=[
        'Topic :: Software Development',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
      entry_points={
          'console_scripts': ['bastet = src.main:main']
      },
      install_requires=INSTALL_REQUIREMENTS,
      tests_require=["pytest", ],
      package_data={},
      include_package_data=True,
      cmdClass={
          'install': InstallCommand
      })
