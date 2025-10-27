from setuptools import setup, find_packages

setup(
    name="rtcfewsio",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["fewsxml", "numpy", "rtctools"],
    author="Farid Alavi",
    author_email="farid.alavi@deltares.nl",
    description="A library that facilitates the input/output data exchange between RTC-Tools and Delft FEWS.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.com/FaridAlavi/rtcfewsio",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)


