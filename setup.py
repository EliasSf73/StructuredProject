from setuptools import find_packages,setup
from typing import List

hypen_e_dot='-e .'
def get_requirements(filename:str)->List[str]:
    'this will return list of requirements'
    requirements=[]
    with open(filename,'r') as file:
        requirements=file.readlines()
            # remove any leading or trailing white space
        requirements=[req.strip() for req in requirements]
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements




setup(
name='MLPROJECT',
version='0.0.1',
author='elias',
author_email='eliassfirisa@gmail.com',
install_requires=get_requirements('requirements.txt')
description="This is an ML Project"
)