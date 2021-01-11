import pathlib

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="easytorch",
    version="2.0.7",
    description="Easy Neural Network Experiments with pytorch",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/sraashis/easytorch",
    author="Aashis Khana1",
    author_email="sraashis@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['easytorch', 'easytorch.config', 'easytorch.data', 'easytorch.metrics', 'easytorch.utils', 'easytorch.vision'],
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'scikit-learn', 'scikit-image',
                      'pillow', 'matplotlib', 'opencv-python', 'pandas', 'seaborn']
)
