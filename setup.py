from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="supertree",
    version="0.5.4",
    description="Visualize decision tree in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mljar/supertree",
    author="MLJAR Sp. z o.o.",
    author_email="contact@mljar.com",
    license="LICENSE.txt",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True,
    python_requires=">=3.7.1",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "visualization",
        "decision-tree",
        "machine-learning",
        "data-analysis",
        "data-mining",
        "classification",
        "regression",
        "mljar",
    ],
)
