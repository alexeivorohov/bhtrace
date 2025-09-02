from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bhtrace",
    version="0.1.0",
    author="Alexei Vorokhov",
    author_email="alexei.vorohov@yandex.ru",
    description="A PyTorch-based library for modeling images of compact objects and ray-tracing in general relativity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexeivorohov/bhtrace",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires='>=3.12',
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'sympy',
    ],
)
