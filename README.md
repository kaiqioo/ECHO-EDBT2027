# ECHO-EDBT2027
Source code and artifacts for ECHO: Equivalence Class-Based Hypergraph Optimizer

This repository contains the implementation of ECHO, an equivalence class-based materialized view selection framework for distributed OLAP query processing.

ECHO builds compact equivalence class (EC) summaries from distributed table partitions, models query reuse with a query--EC hypergraph, and selects useful ECs under a storage budget using reinforcement learning.

The artifact includes the code for data generation, query decomposition, EC construction, hypergraph construction, RL-based EC selection, online evaluation, and baseline comparison.


The code was tested with the following environment:
```text
Python 3.8
PyTorch
NumPy
Pandas
NetworkX
PyYAML
Matplotlib
scikit-learn
Apache Spark 3.5.3
Scala 2.12
```

Install Python dependencies:
```text
pip install -r requirements.txt
```
Or use Conda:
```text
conda env create -f environment.yml
conda activate echo
```

## 2. Repository Structure

```text
echo-artifact/
├── configs/                 # Experiment configuration files
├── echo/
│   ├── data_generation/     # Data and workload generation
│   ├── query_planner/       # Query decomposition and query-cell mapping
│   ├── lattice_builder/     # EC construction and EC matching
│   ├── hypergraph/          # Query--EC hypergraph construction
│   ├── rl_selection/        # RL-based EC selection
│   ├── online_phase/        # Online query evaluation
│   └── utils/               # Utility functions
├── scripts/                 # Running scripts
├── examples/                # Small example data
└── tests/                   # Unit tests
```




