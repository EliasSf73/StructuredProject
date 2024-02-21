from setuptools import find_packages,setup
from typing import List

hypen_e_dot='-e .'
from setuptools import setup

def get_requirements(filename):
    with open(filename, 'r') as file:
        requirements = file.readlines()
    # Remove any whitespace or newline characters
    requirements = [line.strip() for line in requirements]
    return requirements

setup(
    name='MLPROJECT',
    version='0.0.1',
    author='elias',
    author_email='eliassfirisa@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    description="This is an ML Project"
)





