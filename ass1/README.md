# Perceptron

By Erik Zhivkoplias and Jeroen Overschie.

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

## Results

To make our code more modular, our entire implementation was split into multiple components.

After the Perceptron was ran, results are dumped to a file. They can be plotted with `plotting.py`. The program will automatically plot the data with which it is currently configured. In other words; the data generated with the parameters set in `parameters.py`. In order to create multi-line plots, a specialized notebook segment can be run, which is located in `more_plotting.ipynb`.

### Parameters

Change the parameters in `parameters.py`.


## About

University of Groningen (c) 2020