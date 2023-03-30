from setuptools import setup, find_packages

setup(
    name="mi_mmm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "numpy"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple example library",
    #url="https://github.com/yourusername/mylibrary",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
