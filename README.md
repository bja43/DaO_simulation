# DaO_simulation
The DAG adaptation of the Onion method

## Example using DaO in Python

```python
p = 10
ad = 4
n = 100

g = er_dag(p, ad=ad)
sf_out(g)
g = randomize_graph(g)

R, B, O = corr(g)
X = simulate(B, O, n)
X = standardize(X)

df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(p)])
```

## Example using DaO in R

```r
p <- 10
ad <- 4
n <- 100

g <- er_dag(p, ad=ad)
g <- sf_rewire_out(g)
g <- randomize_graph(g)

model <- corr(g)

X <- simulate(model$B, model$O, n)
X <- standardize(X)
```
