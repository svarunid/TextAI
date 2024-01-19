# Contributing 
Contributions are very much welcomed! This repository serves for the purpose of learning and improving. Professionals working in the industry are encouraged to contribute to this repository. 

## Repository Setup
1. Fork the repository.
2. Clone the forked repository to your local machine.
3. Add the upstream remote to your local repository. You may later use this remote to sync your forked repository with the upstream repository.
```bash
git remote add upstream https://github.com/svarunid/tai
```
4. Create a new branch for your contribution. Push your changes to this branch.
```bash
git checkout -b <branch-name>
```
5. Make your changes and commit them.
```bash
git add <file-name>
git commit -m "commit message"
```
6. Push your changes to your forked repository.
```bash
git push origin <branch-name>
```
7. Create a pull request from your forked repository to the upstream repository.

## Setup
1. Clone the repository to your local machine.
```bash
git clone https://github.com/<username>/tai
cd tai
```
2. Create a virtual environment. It is recommended to use [`virtualenv`](https://virtualenv.pypa.io/en/latest/installation.html) or [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for this purpose.
```bash
virtualenv venv
source venv/bin/activate

micromamba create -n venv python=3.11
```
3. Install the python packages in editable mode.
```bash
pip install -e .[dev]
```

## Testing (Optional)
It is highly recommended to write tests for your code. This ensures that the code is working as expected and also helps in debugging. While the main purpose of this repository is to serve as a learning resource, this code in this repository is used to train models for potentially real world applications. Hence, it is important to ensure that the code is working as expected.