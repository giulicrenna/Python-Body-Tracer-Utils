from setuptools import setup, find_packages
from typing import List
import platform

def read_requirements(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

requirements: List[str] = []

if platform.system() == 'Windows':
    windows_requirements = read_requirements('requirements_windows.txt')
    requirements.extend(windows_requirements)

elif platform.system() == 'Linux':
    linux_requirements = read_requirements('requirements_linux.txt')
    requirements.extend(linux_requirements)

setup(
    name='your_package_name',
    version='0.1',
    install_requires=requirements,
    packages=find_packages()
)
