import io
from setuptools import find_packages, setup


# Read in the README for the long description on PyPI
def long_description():
    with io.open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()
    return readme


setup(
    name="taxocrproject",
    version="0.1",
    description="practice python deep-learning taxocrproject",
    long_description=long_description(),
    url="https://github.com/sungbeom90/taxocrproject",
    author="samuel",
    author_email="sungbeom90@gmail.com",
    license="Kosa",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8.5"
        "Programming Language :: Python :: 3.9.5",
    ],
    zip_safe=False,
)
