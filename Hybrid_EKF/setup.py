from setuptools import setup, find_packages

setup(
    name="lorenz_hybrid_ekf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
    ],
) 