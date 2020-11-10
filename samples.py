import numpy as np


sevenlayer = {
    'c': np.array([3.81, 20.08, 1.20, 0.07, 0.002]),
    'm': np.array([19.20, 409.60, 4816.90, 294.91, 7.68]),
    'K': np.array([50.18, 12.54, 1.54, 0.77, 0.04])
}

sevenlayergc = {
    'c': np.array([3.81, 4.89, 20.08, 1.20, 0.07, 0.002]),
    'm': np.array([19.20, 19570.18, 409.60, 4816.90, 294.91, 7.68]),
    'K': np.array([50.18, 50.18, 12.54, 1.54, 0.77, 0.04]),
    'p': np.array([0.99, 0.99, 0.01, 0.01, 0.01, 0.01]),
    'g': np.array([0, 0.99, 0, 0, 0, 0.01])
}

alexnet = {
    'c': np.array([105.73, 224.34, 149.52, 112.14, 74.84, 37.75, 16.78, 4.10]),
    'm': np.array([139.78, 1229.82, 3540.48, 2655.74, 1770.50, 151011.39,
                   67125.25, 16388.00]),
    'K': np.array([279.94, 173.06, 259.58, 259.58, 36.86, 16.38, 16.38, 4.00])
}

alexnet_v2 = {
    'c': np.array([105.73, 224.34, 149.52, 112.14, 74.84, 37.75, 16.78]),
    'm': np.array([139.78, 1229.82, 3540.48, 2655.74, 1770.50, 151011.39,
                   67157.25]),
    'K': np.array([279.94, 173.06, 259.58, 259.58, 36.86, 16.38, 4.00])
}

alexnetgc = {
    'c': np.array([105.73, 224.34, 5.55,
                   149.52, 112.14, 74.84, 37.75, 16.78]),
    'm': np.array([139.78, 1229.82, 22185.22,
                   3540.48, 2655.74, 1770.50, 151011.39, 67157.25]),
    'K': np.array([279.94, 173.06, 173.06,
                   259.58, 259.58, 36.86, 16.38, 4.00]),
    'p': np.array([0.772, 0.772, 0.772,
                   0.228, 0.228, 0.228, 0.228, 0.228]),
    'g': np.array([0, 0, 0.772, 0, 0, 0, 0, 0.228])
}

resnet18 = {
    'c': np.array([]),
    'm': np.array([]),
    'K': np.array([])
}

resnet101 = {
    # Layer 1 = Conv1+Pool1, Layer 2 = Conv2_*, Layer 3 = Conv3_*, Layer 4,5,6,7 = Conv4_* (5+6+6+6), # Layer 8 = Conv5_*
    'c': np.array([118.01, 616.56, 757.86, 950.53, 1156.06, 1156.06, 1156.06, 565.18, 2.20]),
    'm': np.array([37.63, 786.43, 2228.22, 21757.95, 26738.69, 26738.69, 26738.69, 51380.22, 8388.61]),
    'K': np.array([802.82, 802.82, 200.71, 50.18, 50.18, 50.18, 50.18, 12.54, 4.00]),
}

resnet152 = {
    # Layer 1 = Conv1+Pool1, Layer 2 = Conv2_*, Layer 3 = Conv3_*, Layer 4,5,6,7 = Conv4_* (8+8+8+8), # Layer 8 = Conv5_*
    'c': np.array([118.01, 616.56, 1528.56, 1528.56, 1541.41, 1541.41, 1541.41, 565.18, 2.20]),
    'm': np.array([37.63, 786.43, 4587.52, 35127.30, 35651.58, 35651.58, 35651.58, 51380.22, 8388.61]),
    'K': np.array([802.82, 802.82, 200.71, 50.18, 50.18, 50.18, 50.18, 12.54, 4.00]),
}

cnns = {
    '7layer': sevenlayer,
    '7layergc': sevenlayergc,
    'alex': alexnet,
    'alexcatdog': alexnet_v2,
    'alexgc': alexnetgc,
    'resnet18': resnet18,
    'resnet101': resnet101,
    'resnet152': resnet152,
}


nodes = {
    'stm': {'m': 512, 'c': 1000000, 'e': 40},
    'pizero': {'m': 256 * 1024, 'c': 1000000, 'e': 100},
    'pi3': {'m': 512 * 1024, 'c': 1000000, 'e': 560},
    'odroid': {'m': 1024 * 1024, 'c': 1000000, 'e': 307.2},
    'jetson': {'m': 4 * 1024 * 1024, 'c': 1000000, 'e': 5000},
    'orangepizero': {'m': 128 * 1024, 'c': 1000000, 'e': 480},
    'beagleboneai': {'m': 512 * 1024, 'c': 1000000, 'e': 300}
}
node_keys = ['stm', 'pizero', 'pi3', 'odroid', 'jetson']


def get_cnn_configuration(cnnname):
    cnn = cnns[cnnname]
    if 'p' in cnn:
        return cnn['K'], cnn['m'], cnn['c'], cnn['p'], cnn['g']
    return cnn['K'], cnn['m'], cnn['c']


def get_cnn_configuration_c(cnnname):
    cnn = cnns[cnnname]
    if 'p' in cnn:
        p = cnn['p']
        g = cnn['g']
    else:
        # Create fake p and g for non EX-CNNs
        p = np.ones(len(cnn['m']))
        g = np.zeros(len(p))
        g[-1] = 1
    return cnn['K'], cnn['m'], cnn['c'], p, g


def get_random_node_config(numberofnodes):
    idxs = np.random.choice(len(node_keys), numberofnodes, replace=False)
    m = np.zeros(numberofnodes)
    c = np.zeros(numberofnodes)
    e = np.zeros(numberofnodes)
    for i, nodeidx in enumerate(idxs):
        m[i] = nodes[node_keys[nodeidx]]['m']
        c[i] = nodes[node_keys[nodeidx]]['c']
        e[i] = nodes[node_keys[nodeidx]]['e']

    return {'m': m, 'c': c, 'e': e}


def get_node_configuration(confname):
    node_configs = {
        'random1': get_random_node_config(1),
        'random2': get_random_node_config(2),
        'random3': get_random_node_config(3),
        'stm_pi0': {'m': np.array([nodes['stm']['m'], nodes['pizero']['m']]),
                    'c': np.array([nodes['stm']['c'], nodes['pizero']['c']]),
                    'e': np.array([nodes['stm']['e'], nodes['pizero']['e']])},
        'stm_odroid': {'m': np.array([nodes['stm']['m'], nodes['odroid']['m']]),
                       'c': np.array([nodes['stm']['c'], nodes['odroid']['c']]),
                       'e': np.array([nodes['stm']['e'], nodes['odroid']['e']])},
        'stm_pi3': {'m': np.array([nodes['stm']['m'], nodes['pi3']['m']]),
                    'c': np.array([nodes['stm']['c'], nodes['pi3']['c']]),
                    'e': np.array([nodes['stm']['e'], nodes['pi3']['e']])},
        'orange_beagle': {'m': np.array([nodes['orangepizero']['m'], nodes['beagleboneai']['m']]),
                          'c': np.array([nodes['orangepizero']['c'], nodes['beagleboneai']['c']]),
                          'e': np.array([nodes['orangepizero']['e'], nodes['beagleboneai']['e']])},
        'orange_beagle_pi3': {'m': np.array([nodes['orangepizero']['m'], nodes['beagleboneai']['m'], nodes['pi3']['m']]),
                              'c': np.array([nodes['orangepizero']['c'], nodes['beagleboneai']['c'], nodes['pi3']['c']]),
                              'e': np.array([nodes['orangepizero']['e'], nodes['beagleboneai']['e'], nodes['pi3']['e']])},
    }
    return node_configs[confname]['m'], node_configs[confname]['c'], \
        node_configs[confname]['e']


def get_sequence_of_nodes(confname, probabilities, num_nodes):
    configuration = get_node_configuration(confname)
    num_different_nodes = len(probabilities)
    sequence = np.random.choice(
        np.arange(num_different_nodes), size=num_nodes,
        replace=True,
        p=probabilities
    )
    return {'m': np.array([configuration['m'][i] for i in sequence]),
            'c': np.array([configuration['c'][i] for i in sequence]),
            'e': np.array([configuration['e'][i] for i in sequence])}
