# DaOsim

The DAG adaptation of the Onion method


## Example Usage

```python
import pandas as pd

from daosim import er_dag            # generates a Erdos-Renyi directed acyclic graph
from daosim import sf_out            # rewires a DAG to have scale-free out degree
from daosim import randomize_graph   # randomly shuffles the order of the variables
from daosim import corr              # samples a correlation matrix 
from daosim import simulate          # simulates a dataset
from daosim import standardize       # standardizes a dataset

p = 10    # number of variables
ad = 4    # average degree
n = 100   # number of samples

g = er_dag(p, ad=ad)
g = sf_out(g)
g = randomize_graph(g)

R, B, O = corr(g)
X = simulate(B, O, n)
X = standardize(X)

cols = [f"X{i + 1}" for i in range(p)]
df = pd.DataFrame(X, columns=cols)
