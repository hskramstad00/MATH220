# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:45:16 2026

@author: gebo
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.sparse import load_npz


# Laster inn matrise
A_sparse=load_npz('NMBU_adjacency_matrix.npz')

# Lager en dict for som gir url-en for hver matrise.
with open("adj_index.json", 'r') as f:
    index_to_url = json.load(f)


# Omgjør en sparse matrise til "vanlig" matrise A
A= A_sparse.toarray()
