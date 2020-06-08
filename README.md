# Tindar: end-to-end Operations Research

Welcome to the Tindar project: the optimal matchmaker! After some hard work, the Tindar project is finally
deployed. Visit the Tindar web application [here](http://tindar-engine-xs-chx6ixua2q-ew.a.run.app)!

## Introduction
The main purpose of this project is to build and share knowledge on how to develop
an end-to-end data product. This personally helps me to let some concepts sink in
and is a gateway for others to give feedback on my working style (and possibly pick up
some concepts if they were not yet familiar).

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

## Installation
I would recommend to isolate the dependencies of this project in a virtual environment with [venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Clone and install dependencies
For now, you can clone this repository and experiment by importing the modules manually. Go to a directory of your choice, then clone the repository:
```
git clone https://github.com/lennart-damen/tindar
```
Create a virtual environment named "env":
```
python -m venv env
```
Activate the environment (example below for Windows):
```
cd env
cd Scripts
activate
```
Go back to the home folder
```
cd ..
cd ..
```
Install the dependencies
```
pip install -r requirements.txt
```
for minimal local experimentation with Tindar and Flask. You can also:
```
pip install -r requirements_extend.txt
```
in which case you can launch Jupyter notebooks with all our dependencies after registering your virtual environment (https://janakiev.com/blog/jupyter-virtual-envs/, see Add Virtual Environment to Jupyter Notebook).

## Usage
To get a better grasp on the purpose and capabilities of this project, check out [the web app](http://tindar-engine-xs-chx6ixua2q-ew.a.run.app)and/or read the documentation in ./documentation/report.

First, make sure your virtual environment is still activated and you are in the repository home.

To start the Flask app locally on port 8080:
```
python run.py
```
You can also investigate how the PuLP solver compares to the heuristic, by doing a Tindar experiment
```
cd tindar-engine
python tindar.py
```

## Project Organization
------------

    ├── .gcloudignore                   <------ Specifies build context for GCP cloud build
    ├── .gitignore
    ├── code_evolution                  <------ Step-by-step development of Tindar sourcecode
    ├── cloudbuild.yml                  <------ To build and push Docker image to GCP container registry
    ├── data                            <------ Folder to hold results of running "tindar-engine/tindar.py"
    ├── Dockerfile
    ├── documentation                   <------ Eplanatory text documents
    ├── flask_application               <------ The self-documented Tindar web application
    │   ├── templates                   <------ Folder with HTML
    │   ├── __init__.py                 <------ HTML
    │   └── views.py                    <------ Routes
    ├── LICENSE
    ├── models                          <------ The BiLP models
    ├── notebooks
    │   └── Tindar_model.ipynb          <------ Development of optimal Tindar solution
    ├── README.md
    ├── requirements_model_extend.txt   <------ Environment incl. jupyter integration and linting
    ├── requirements_model.txt          <------ Minimal environment for the Tindar project
    ├── run.py                          <------ Entrypoint for Flask (only for local development)
    ├── setup.py
    ├── tindar-engine                   <------ Tindar sourcecode
    │   │── tindar.py
    │   │── custom_timer.py
    ├── tests                           <------ Testing sourcecode with pytest

