import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tindar-engine",
    version="1.1.2",
    author="Lennart Damen",
    author_email="lennartdmn@gmail.com",
    description="Create and solve Tindar problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lennart-damen/tindar",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
