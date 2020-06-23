from distutils.core import setup

setup(
    name='webapp-backend',
    version='0.1dev',
    packages=['backend',],
    license='unlicense',
    install_requires=[
        'Flask',
        'gunicorn',
    ],
    long_description=open('README.txt').read(),
)
