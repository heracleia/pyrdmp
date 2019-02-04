from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()

setup(name='pyrdmp',
      version='0.1.0',
      description='Python Library for Reinforced Dynamic Movement Primitives',
      url='https://github.com/Heracleia/pyrdmp',
      author='Michail Theofanidis, Joe Cloud, James Brady',
      author_email='git@joe.cloud',
      license=license,
      packages=find_packages(exclude=('examples'))
)