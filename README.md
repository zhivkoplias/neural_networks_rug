# Perceptron training assignment
Neural Networks course RUG

## Usage

Run 

```shell
python3 ./perceptron.py
```

In order to utilize multi-core processing use:

```shell

python3 ./perceptron.py --multicore
```

Do note VSCode debugging is not available when using multi-core processing.

### Using virtual env (instead of anaconda)
To install dependencies:
1. `python3 -m venv ./venv`
2. `source ./venv/bin/activate`
3. `pip3 install -r requirements.txt`

In order to add a new dependency:
1. `pip3 install <dep>`
2. `pip3 freeze > requirements.txt`