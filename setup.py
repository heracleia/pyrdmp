from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='pyrdmp',
      version='0.2.2',
      description='Python Library for Reinforced Dynamic Movement Primitives',
      url='https://github.com/Heracleia/pyrdmp',
      author='Michail Theofanidis, Joe Cloud, James Brady',
      author_email='git@joe.cloud',
      license=license,
      packages=find_packages(exclude=('examples')),
      install_requires=required
)
