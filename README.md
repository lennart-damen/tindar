# Tindar: end-to-end Operations Research

Welcome to the Tindar project: the optimal matchmaker!

## Introduction
The main purpose of this project is to share my knowledge on how to build
an end-to-end data product, both for myself to let some concepts sink in
and for others to give feedback on my working style (possibly pick up some concepts
if they were not yet familiar).

I love working on Data Science, Machine Learning, and Operations Research problems,
but I also see around me that many projects never really lift off. I use the Tindar project
to prove to myself and others that the technical aspect should not be the
limiting factor.

## Tindar challenge
Suppose you owned a dating company. Once every week, people come to your venue to find their
ideal partner. During the speed-dating round, every person expresses if he/she would like
to meet his/her date in real-life in the next week. Your people have busy schedules,
so they can only go on a real date with a single person.

Your job is to make your community as happy as possible by pairing up as many users as
you can.

## Tindar components
The Tindar project consists of three components:

1. the Tindar engine: generate and solve tindar problems (this repo)
2. the Tindar API: interact with the Tindar engine (TODO: href here)
3. the Tindar game: interactive game to show users the power of Operations Research (TODO: href here)

Done:
- The mathematical Tindar model
- The Tindar engine
    - using OOP to solve games with PuLP and heuristic
    - using OOP to generate games

To do:
- The Tindar API:
    - build Flask app
    - deploy to AWS Lambda/GCP functions
- The Tindar game
    - Build user Interface with Dash
    - Deploy to Cloud Platform

## Project Organization
------------

    ├── LICENSE
    ├── README.md
    ├── .gitignore
    ├── code_evolution                  <------ Step-by-step development of sourcecode
    ├── documentation                   <------ Eplanatory text documents
    ├── models                          <------ The BiLP models
    ├── notebooks
    │   └── Tindar_model.ipynb          <------ Development of optimal Tindar solution
    ├── requirements_model_extend.txt   <------ Environment incl. jupyter integration and linting
    ├── requirements_model.txt          <------ Minimal environment for the Tindar model
    ├── setup.py
    ├── src
    │   │── tindar.py
    ├── tests                           <------ Testing sourcecode with pytest


## Installation
### Clone and use
For now, you can clone this repository and experiment by importing the modules manually. Make sure that the Python interpreter can find the paths the the modules.

### PyPi distribution
I am still working on a PyPi distribution, which is not ready yet. Once ready, you will be able to install with pip.

I would recommend to isolate the dependencies of this project in a virtual environment with [venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After activating the environment, you can install the Tindar engine from test.pypi:

```
pip install -i https://test.pypi.org/simple/ tindar-engine
```

Also make sure to install the additional dependencies listed in this repository. Download requirements.txt and:
```
pip install -r requirements.txt
```

This package is just experimental, and I prefer to keep my project structure as defined above. To use the package, start an interpreter. Then, you can use the package like so:

```python
from src.tindar import Tindar
```

TO DO:
- code above does not work, get error: from timer import Timer -> ModuleNotFoundError: No module named 'timer'
- Tindar.write_problem() default path will not work in the PyPi distribution