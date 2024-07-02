import numpy as np
import pandas as pd

import sys
import time

sys.path.append("..")

from dao import er_dag, sf_in, sf_out, randomize_graph, corr, cov, simulate, standardize
from timeit import default_timer as timer

import jpype
import jpype.imports

jpype.startJVM("-Xmx4g", classpath="tetrad-current.jar")

import translate

import java.util as util
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.algcomparison as ta


def construct_graph(g, nodes, cpdag=True):
    graph = tg.EdgeListGraph(nodes)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if g[i, j]: graph.addDirectedEdge(b, a)
    if cpdag: graph = tg.GraphTransforms.cpdagForDag(graph)
    return graph


def run_sim(n, p, ad, sf, rep):

    g = er_dag(p, ad=ad)
    if sf[0]: g = sf_out(g)
    if sf[1]: g = sf_in(g)
    g = randomize_graph(g)

    _, B, O = corr(g)
    # _, B, O = cov(g)
    # _, B, O = cov(g, lb_b=0.5, ub_b=2, lb_o=1, ub_o=1)

    gaus_err = lambda *x: np.random.normal(0, np.sqrt(x[0]), x[1])
    # gumb_err = lambda *x: np.random.gumbel(0, np.sqrt(6.0 / np.pi**2 * x[0]), x[1])
    # exp_err = lambda *x: np.random.exponential(np.sqrt(x[0]), x[1])
    # lapl_err = lambda *x: np.random.laplace(0, np.sqrt(x[0] / 2), x[1])
    # unif_err = lambda *x: np.random.uniform(-np.sqrt(3 * x[0]), np.sqrt(3 * x[0]), x[1])

    X = simulate(B, O, n, gaus_err)
    if standardize_data: X = standardize(X)

    df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(p)])
    data = translate.df_to_data(df)
    nodes = data.getVariables()
    cpdag = construct_graph(g, nodes)

    score = ts.score.SemBicScore(data, True)
    score.setPenaltyDiscount(penalty_discount)
    score.setStructurePrior(0)

    test = ts.test.IndTestFisherZ(data, alpha)

    algs = []
    graphs = []
    times = [timer()]

    run_boss = True
    run_grasp = False
    run_fges = True
    run_pc = True

    if run_boss:
        # print("boss")
        algs.append("boss")
        boss = ts.Boss(score)
        boss.setUseBes(boss_bes)
        boss.setNumStarts(boss_starts)
        boss.setNumThreads(boss_threads)
        boss.setUseDataOrder(False)
        boss.setResetAfterBM(True)
        boss.setResetAfterRS(False)
        boss.setVerbose(False)
        boss = ts.PermutationSearch(boss)
        graphs.append(boss.search())
        times.append(timer())

    if run_grasp:
        # print("grasp")
        algs.append("grasp")
        grasp = ts.Grasp(score)
        grasp.setAllowInternalRandomness(True)
        grasp.setDepth(grasp_depth)
        grasp.setNonSingularDepth(1)
        grasp.setNumStarts(grasp_starts)
        grasp.setOrdered(False)
        grasp.setUncoveredDepth(1)
        grasp.setUseDataOrder(False)
        grasp.setUseRaskuttiUhler(False)
        grasp.setUseScore(True)
        grasp.setVerbose(False)
        grasp.bestOrder(util.ArrayList(nodes))
        graphs.append(grasp.getGraph(True))
        times.append(timer())

    if run_fges:
        # print("fges")
        algs.append("fges")
        graphs.append(ts.Fges(score).search())
        times.append(timer())

    if run_pc:
        # print("pc")
        algs.append("pc")
        pc = ts.Pc(test)
        pc.setDepth(pc_depth)
        pc.setStable(pc_stable)
        graphs.append(pc.search())
        times.append(timer())

    return (tg.GraphUtils.replaceNodes(cpdag, nodes), data,
            [(alg, tg.GraphUtils.replaceNodes(graphs[i], nodes),
              times[i + 1] - times[i]) for i, alg in enumerate(algs)])


reps = 10

unique_sims = [(n, p, ad, sf)
               for n in [200, 1_000]
               for p in [20, 60]
               for ad in [5, 10]
               for sf in [(1, 0)]]

standardize_data = True

stats = [ta.statistic.AdjacencyPrecision(),
         ta.statistic.AdjacencyRecall(),
         ta.statistic.OrientationPrecision(),
         ta.statistic.OrientationRecall()]

penalty_discount = 2
alpha = 0.01

boss_bes = False
boss_starts = 10
boss_threads = 8

grasp_starts = 10
grasp_depth = 3

pc_depth = -1
pc_stable = True


results = []

for n, p, ad, sf, rep in [(*sim, rep) for sim in unique_sims for rep in range(reps)]:
    t = time.strftime("%H:%M:%S", time.localtime())
    print(f"time: {t} | samples: {n} | variables: {p} | avg_deg: {ad} | scale-free: {sf} | rep: {rep}")
    true_cpdag, data, algs = run_sim(n, p, ad, sf, rep)
    for alg, est_cpdag, seconds in algs:
        results.append([n, p, ad, sf, alg, rep] + [stats[i].getValue(true_cpdag, est_cpdag, data) for i in range(len(stats))] + [seconds])

param_cols = ["samples", "variables", "avg_deg", "scale-free", "algorithm", "run"]
stat_cols = ["adj_pre", "adj_rec", "ori_pre", "ori_rec", "seconds"]
df = pd.DataFrame(np.array(results), columns=param_cols+stat_cols)
for col in stat_cols: df[col] = df[col].astype(float)

param_cols.remove("run")
print(f"\n\nreps: {reps}\n")
print(df.groupby(param_cols)[stat_cols].agg("mean").round(2).to_string())
