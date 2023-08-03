#%%
# from init_rcParams import set_mpl_settings
# set_mpl_settings()
#%%
import pandas as pd
import networkx as nx
import numpy as np
# %%
import pickle
import gravis as gv

from pathlib import Path
data_folder = Path.home()/'Documents/'

with open(data_folder/'treasure/data_files/main/df_list_nov.pkl', 'rb') as fp:
    df_list = pickle.load(fp)
    
for i, df in enumerate(df_list):
    # Only take adults
    df = df.loc[df.age >= 21]
    # Only take where we know drinking data
    df = df.dropna(subset = ['d_state_ego','d_state_alter'], axis = 0)
    # Also drop where drinking state = -1 because we do not have sex
    df = df.loc[df.d_state_ego != -1]
    df_list[i] = df

df = df_list[2]
#%%
#%%
G = nx.from_pandas_edgelist(df,'shareid','sharealterid',['ALTERTYPE'])
nx.set_node_attributes(G, df.loc[:,['shareid', 'dpw','age', 'd_state_ego']].groupby('shareid').mean().to_dict('index'))
nx.set_node_attributes(G, df.loc[:,['sharealterid', 'dpw_alter','alter_age', 'd_state_alter']].groupby('sharealterid').mean().rename(columns  = {'dpw_alter':'dpw', 'alter_age':'age', 'd_state_alter':'d_state_ego'}).to_dict('index'))
#%%
# RELABEL ALL NODES SO THEIR NUMBERING IS 1toN
mapping = dict(zip(G, range(0, G.number_of_nodes())))
G_relabled = nx.relabel_nodes(G, mapping)

# %%
# SELECT LARGEST CONNECTED CLUSTER
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])
#%%
# RELABEL ALL NODES SO THEIR NUMBERING IS 1toN
mapping = dict(zip(G0, range(0, G0.number_of_nodes())))
G0_relabled = nx.relabel_nodes(G0, mapping)

G3 = G.subgraph(Gcc[3])
mapping = dict(zip(G3, range(0, G3.number_of_nodes())))
G3_relabled = nx.relabel_nodes(G3, mapping)

G1 = G.subgraph(Gcc[1])
mapping = dict(zip(G1, range(0, G1.number_of_nodes())))
G1_relabled = nx.relabel_nodes(G1, mapping)
# # %%
# gv.d3(G0)
# # %%
# len(G0)
# # %%
#%%
# %%
# nx.get_node_attributes(G, 'd_state_ego')

# %%
