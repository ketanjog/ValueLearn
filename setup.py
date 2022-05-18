import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ValueLearn",
    version="0.1-dev",
    author="Ketan Jog",
    author_email="kj2473@columbia.edu",
    description="Codebase to experiment with different environment structures to see which ones favor MB/MF/SR learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ketanjog/ValueLearn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
