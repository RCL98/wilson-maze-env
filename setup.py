from setuptools import find_packages, setup

setup(
    name="wilson-maze-env",
    version="2.3.0",
    install_requires=["gymnasium==0.29.1", "pygame==2.5.2"],
    packages=find_packages()
)