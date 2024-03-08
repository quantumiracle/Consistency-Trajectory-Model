from setuptools import setup, find_packages

setup(
    name="CTM",
    version="0.1",
    description="Minimal unofficial implementation of Consistency Trajectory models in pytorch",
    license="MIT",
    author="quantumiracle",
    url="https://github.com/quantumiracle/Consistency-Trajectory-Model",
    packages=find_packages(),
    install_requires=[
        "torch",
        "einops",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "numpy"
    ]
)
