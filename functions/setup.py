from setuptools import setup, find_packages

setup(
    name='MC-TIRE',
    version='1.0',
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