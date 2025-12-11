## Installation and Running

### Prerequisites and Installation

The package requires Python 3.8 or higher. To install the package and its dependencies, run:

```bash
$ git clone https://github.com/fatihdogangun/sym_learning.git
$ cd symbolic_operator_learning
$ pip install -r requirements.txt
```

The planning evaluation requires Fast Downward, a classical AI planner. To install the planning system, run:

```bash
$ git clone https://github.com/aibasel/downward.git
$ cd downward
$ python build.py
```

Detailed instructions for the installation of the planning system are available at the [Official Repository](https://github.com/aibasel/downward) of the planner.

### Running

#### 1. Data Collection

```bash
$ cd scripts
$ python collect_mp.py -s explore.py -d ../data/<dataset_name> \
                       -N 12500 -T 50 -p 8 -n_min 2 -n_max 4
```

#### 2. Model Training

```bash
$ python train.py -c config.yaml
```

#### 3. Operator Extraction

```bash
$ python learn_rules.py -n <model_name>
```

#### 4. Evaluate Planning

```bash
$ python test.py -n <model_name> -o 3 -t 100 -k 5 -p 8 --save_images
```
