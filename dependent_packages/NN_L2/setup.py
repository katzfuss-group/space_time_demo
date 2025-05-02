from setuptools import setup, find_packages
from setuptools import Extension

setup(
    name="NN_L2",
    version="0.0.0",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "scikit-learn"],
    extras_require={"faiss": ["faiss"]}
)
