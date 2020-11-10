import numpy as np
from matplotlib import pyplot as plt

import os


def euclidean_distance(x1, x2, y1, y2):
    x = x1 - x2
    y = y1 - y2
    return np.sqrt(x * x + y * y)


def create_graph(nodes=10):
    g = dict()
    for i in range(nodes):
        g[i] = set()
    return g


def add_node_to_graph(graph):
    n = len(graph)
    graph[n] = set()
    return graph


def generate_random_positions(nodes, minx=-10, maxx=10, miny=-10, maxy=10):
    xl = maxx - minx
    yl = maxy - miny
    x = np.zeros(0)
    y = np.zeros(0)
    for i in range(nodes):
        x = np.append(x, np.random.uniform(0, 1, 1) * xl + minx)
        y = np.append(y, np.random.uniform(0, 1, 1) * yl + miny)

    return x, y


def add_arcs_with_max_distance(graph, xpos, ypos, dmax=2.5):
    # Add the arc (i,j) iff d_e(i,j) < dmax, where d_e is the Euclidean distance
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if euclidean_distance(xpos[i], xpos[j], ypos[i], ypos[j]) < dmax:
                # Add both the arc (i,j) and (j,i) since the graph is acyclic.
                graph[i].add(j)
                graph[j].add(i)
    return graph


def find_all_paths(graph, a, b, path=list(), already_visited_nodes=set()):
    """
    Recursively find all paths starting from a and reaching b.
    """
    path = path + [a]

    # Check for the end of recursion
    if a == b:
        # print('*** !!! Find Path {} !!! ***'.format(path))
        return [path]

    # Check if a is isolated (in that case return empty list)
    if not len(graph[a]):
        return list()

    # Create the list of paths
    paths = list()
    for node in graph[a]:
        # Check if the node has been already visited to avoid infinite
        # recursion due to cycles.
        if not len(already_visited_nodes) or node not in already_visited_nodes:
            if not len(already_visited_nodes):
                avn = {a}
            else:
                avn = already_visited_nodes
                avn.add(a)
            recursive_paths = find_all_paths(graph, node, b, path=path,
                                             already_visited_nodes=avn)
            for p in recursive_paths:
                paths.append(p)

    return paths


def find_min_distance(graph, a, b):
    paths = find_all_paths(graph, a, b)
    if not len(paths):
        return np.inf
    return min({len(p) for p in paths}) - 1


def plot(graph, xpos, ypos, xm=5, outdir='.', figname=None, figsize=(15, 10),
         source_available=True, sink_different_from_source=False,
         num_sources=1, annotate=False,
         color=None, marker=None, size=None, sequence=None):
    # Compute useful stuff
    num_nodes = len(xpos) - num_sources * source_available \
        - num_sources * source_available * sink_different_from_source

    plt.figure(figsize=figsize)
    # Get axes reference to number nodes
    ax = plt.gca()
    plot_nodes(ax, xpos, ypos, xm,
               source_available, sink_different_from_source,
               num_sources, annotate,
               color, marker, size, sequence)
    
    # Compute the arcs
    for k in graph:
        for d in graph[k]:
            if k >= num_nodes:
                # The arc is from the source or to the sink.
                ax.plot([xpos[k], xpos[d]], [ypos[k], ypos[d]],
                        linewidth=0.5, c='orange')
            elif d > k:
                ax.plot([xpos[k], xpos[d]], [ypos[k], ypos[d]],
                    linewidth=0.25, linestyle='--', c='darkblue')
    # Save the figure
    if figname is None:
        plt.show()
    else:
        plt.savefig(os.path.join(outdir, figname))
        plt.close('all')


def path_plot(xpos, ypos, path, xm=5, outdir='./', figname=None, figsize=(15, 10),
              source_available=True, sink_different_from_source=False,
              num_sources=1, path_source_idx=0,
              annotate=False, color=None, marker=None, size=None, sequence=None,
              path_color='darkred', out_prob=None, ax=None, plot_background=True):
    # Compute useful stuff
    num_nodes = len(xpos) - num_sources * source_available \
        - num_sources * source_available * sink_different_from_source

    if ax is None:
        plt.figure(figsize=figsize)
        # Get axes reference to number nodes
        ax = plt.gca()
    if plot_background:
        plot_nodes(ax, xpos, ypos, xm,
                   source_available, sink_different_from_source,
                   num_sources, annotate,
                   color, marker, size, sequence)
    
    # Add source and sink to the path
    if source_available and sink_different_from_source:
      
        # Save source idx in the path
        path = np.append(-2*num_sources+path_source_idx, path)
    elif source_available:
        # Save source idx in the path
        path = np.append(-num_sources+path_source_idx, path)
    # Sink idx in the path
    path = np.append(path, -num_sources+path_source_idx)

    # Plot all the arcs
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        if out_prob is None or i < len(path) - 2:
            plot_arc(ax, x0=xpos[a], x1=xpos[b],
                     y0=ypos[a], y1=ypos[b],
                     color=path_color)

        if out_prob is not None and i > 0 and out_prob[i-1] > 0:
            plot_arc(ax, x0=xpos[a], x1=xpos[-num_sources+path_source_idx],
                     y0=ypos[a], y1=ypos[-num_sources+path_source_idx],
                     color=path_color, ls='-.',
                     annotate='{:.3f}'.format(out_prob[i-1]))
    # Save the figure
    if figname is None:
        pass #plt.show()
    else:
        plt.savefig(os.path.join(outdir, figname))
        plt.close('all')
    
    return ax
    

def plot_nodes(ax, xpos, ypos, xm=5,
               source_available=True, sink_different_from_source=False,
               num_sources=1, annotate=False,
               color=None, marker=None, size=None, sequence=None):
    # Compute useful stuff
    num_nodes = len(xpos) - num_sources * source_available \
        - num_sources * source_available * sink_different_from_source

    ax.set_xlim([-xm, xm])
    ax.set_ylim([-xm, xm])
    # Plot the nodes and annotate them (the last coordinate is the source)
    if sequence is None:
        ax.scatter(xpos[:num_nodes], ypos[:num_nodes], c='darkblue', marker='O', s=50)
    else:
        # Both markers and colors are different from None
        for idx, value in enumerate(sequence):
            ax.scatter(xpos[idx], ypos[idx], c=color[value], marker=marker[value], s=size[value])
    if annotate:
        for i in range(num_nodes):
            ax.annotate(str(i + 1), (xpos[i], ypos[i]))

    # Plot the source
    if source_available and sink_different_from_source:
        ax.scatter(
            xpos[-2*num_sources:-num_sources],
            ypos[-2*num_sources:-num_sources],
            c='orange', s=200, marker='*'
        )
        for i in range(num_sources):
            ax.annotate('S{}'.format(i+1), (xpos[-2*num_sources+i], ypos[-2*num_sources+i]))
        ax.scatter(
            xpos[-1*num_sources:],
            ypos[-1*num_sources:],
            c='darkred', s=200, marker='*'
        )
        for i in range(num_sources):
            ax.annotate('Sink {}'.format(i+1), (xpos[-num_sources+i], ypos[-num_sources+i]))

    elif source_available:
        ax.scatter(
            xpos[-1*num_sources:],
            ypos[-1*num_sources:],
            c='orange', s=200, marker='*'
        )
        for i in range(num_sources):
            ax.annotate('S{}'.format(i+1), (xpos[-num_sources+i], ypos[-num_sources+i]))


def plot_arc(ax, x0, x1, y0, y1, color, lw=1, ls='--', annotate=None):
    ax.plot([x0, x1], [y0, y1],
            linewidth=lw, linestyle=ls, color=color)
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    dx = (x1 - x0) / 10
    dy = (y1 - y0) / 10
    ax.arrow(xm, ym, dx, dy, head_width=0.2, head_length=0.2, fc=color, ec='white')
    if annotate is not None:
        ax.annotate(annotate, (xm, ym))
