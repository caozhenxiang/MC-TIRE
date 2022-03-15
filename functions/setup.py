from setuptools import setup, find_packages
import os

path = os.path.abspath(os.path.join(os.getcwd(), ".."))
with open(os.path.join(path, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='MC-TIRE',
    version='1.0',
    author="Zhenxiang Cao",
    author_email="zhenxiang.cao@esat.kuleuven.be",
    url="https://github.com/caozhenxiang/MC-TIRE",
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    setup_requires=[
        'setuptools',
    ],
    install_requires=[
        'tensorflow',
        'tensorflow_probability',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'tensorflow-addons',
    ]
)