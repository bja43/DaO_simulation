import numpy as np

import java.util as util
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as tg


def df_to_data(df, cont=np.inexact):
    cols = df.columns
    discrete_cols = [col for col in cols if not np.issubdtype(df[col].dtype, cont)]
    category_map = {col: {val: i for i, val in enumerate(df[col].unique())} for col in discrete_cols}
    df = df.replace(category_map)
    values = df.values
    n, p = df.shape

    variables = util.ArrayList()
    for col in cols:
        if col in discrete_cols:
            categories = util.ArrayList()
            for category in category_map[col]:
                categories.add(str(category))
            variables.add(td.DiscreteVariable(str(col), categories))
        else:
            variables.add(td.ContinuousVariable(str(col)))

    if len(discrete_cols) == len(cols):
        databox = td.IntDataBox(n, p)
    elif len(discrete_cols) == 0:
        databox = td.DoubleDataBox(n, p)
    else:
        databox = td.MixedDataBox(variables, n)

    for col, var in enumerate(values.T):
        for row, val in enumerate(var):
            databox.set(row, col, val)

    return td.BoxDataSet(databox, variables)


def mat_to_graph(g, nodes, cpdag=True):
    graph = tg.EdgeListGraph(nodes)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if g[i, j]: graph.addDirectedEdge(b, a)

    if cpdag: graph = tg.GraphTransforms.cpdagForDag(graph)

    return graph
