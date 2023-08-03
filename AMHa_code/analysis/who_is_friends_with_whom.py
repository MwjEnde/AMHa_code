#%%
from cmath import exp
from math import factorial
import numpy as np
import pandas as pd
import networkx as nx
import init_rcParams

init_rcParams.set_mpl_settings()
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive, fixed, interact_manual,widgets

import pickle

from pathlib import Path
# data_folder = Path.home()/'Downloads/sratoolkit.2.11.1-ubuntu64/bin/dbGaP-27276/'
#%%
with open('/home/maarten/Documents/treasure/data_files/main/df_list_nov.pkl', 'rb') as fp:
# with open('./data_files/main/df_list_nov.pkl', 'rb') as fp:
    df_list = pickle.load(fp)
    
dynamic_df = pd.read_csv('./data_files/main/df_nov.csv')

# !!!! takes only adults!!!!
for i, df in enumerate(df_list):
    # Only take adults
    df = df.loc[df.age >= 21]
    # Only take where we know drinking data
    df = df.dropna(subset = ['d_state_ego','d_state_alter'], axis = 0)

    df = df.loc[df.d_state_ego != -1]
    df = df.loc[df.d_state_alter != -1]
    df_list[i] = df
#%%


tot_all_waves = np.zeros((3,4))
exp_tot_all_waves = np.zeros((3,4))
ar_list = np.zeros((7,3,4))
exp_list = np.zeros((7,3,4))
perc_list = np.zeros((7,3,3))
exp_perc_list = np.zeros((7,3,3))
avg_num_connections_per_ego_list = []
for w in range(0,len(df_list)):
    df = df_list[w]
    df = df.assign(connections = 1)
    #! for different ages!!!!
    # df = df.loc[df.age <= 40]
    df2 = df.loc[:,['shareid','d_state_ego','d_state_alter','connections']]

    df2.dropna(inplace = True)

    # actual connections per state
    ar = np.empty((3,4))
    perc = np.empty((3,3))
    for s_ego in (0,1,2):
        for s_alter in (0,1,2,3):
            if s_alter == 3:
                ar[s_ego,s_alter] = df2.loc[df2['d_state_ego'] == s_ego].connections.sum()
            else:
                ar[s_ego,s_alter] = df2.loc[(df2['d_state_ego'] == s_ego) & (df2['d_state_alter'] == s_alter)].connections.sum()
                perc[s_ego,s_alter] = df2.loc[(df2['d_state_ego'] == s_ego) & (df2['d_state_alter'] == s_alter)].connections.sum() / df2.loc[df2['d_state_ego'] == s_ego].connections.sum()

    tot_ego_dict = {'state':[0,1,2], 'tot_ego': [df2.loc[df2.d_state_ego == 0].shareid.nunique(), df2.loc[df2.d_state_ego == 1].shareid.nunique(), df2.loc[df2.d_state_ego == 2].shareid.nunique()]}
    tot_ego_df = pd.DataFrame.from_dict(tot_ego_dict)  
    tot_ego_all = df2.shareid.nunique()

    exp_ar = np.empty((3,4))
    exp_perc = np.empty((3,3))
    for s_ego in (0,1,2):
        for s_alter in (0,1,2):
            exp_ar[s_ego,s_alter] = df2.loc[(df2['d_state_alter'] == s_alter)].connections.sum()/ tot_ego_all * tot_ego_df.loc[tot_ego_df['state']== s_ego].values[0,1]
            exp_perc[s_ego,s_alter] = df2.loc[(df2['d_state_alter'] == s_alter)].connections.sum()/ tot_ego_all * tot_ego_df.loc[tot_ego_df['state']== s_ego].values[0,1] / df2.loc[df2['d_state_ego'] == s_ego].connections.sum()
        exp_ar[s_ego, 3] = exp_ar[s_ego].sum()
            
    tot_all_waves = np.add(tot_all_waves, ar)
    exp_tot_all_waves = np.add(exp_tot_all_waves, exp_ar)
    ar_list[w] = ar
    exp_list[w] = exp_ar
    perc_list[w] = perc
    exp_perc_list[w] = exp_perc
    avg_num_connections_per_ego = ar[:,3]/ tot_ego_df.tot_ego.values
    avg_num_connections_per_ego_list.append(avg_num_connections_per_ego)

#%%
import numpy as np

@interact(w = (0,6,1), all_waves = False)
def plot_cluster_heatmap(w, all_waves = False):

    if all_waves: 
        actual_percentage = perc_list.mean(axis =0)
        expected_percentage = exp_perc_list.mean(axis =0)
    else:
        actual_percentage = perc_list[w]
        expected_percentage = exp_perc_list[w]
    harvest = actual_percentage / expected_percentage
    harvest = np.round(harvest, 2)

    # Exclude specific cells
    harvest[0, 1] = np.nan
    harvest[0, 2] = np.nan
    harvest[1, 2] = np.nan

    vegetables = ["abstaining", "moderate", "heavy"]
    farmers = ["abstaining", "moderate", "heavy"]
    fig, ax = plt.subplots()
    im = ax.imshow(harvest, cmap="Greens")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Get the colormap
    cmap = plt.cm.Greens

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            if np.isnan(harvest[i, j]):
                continue  # Skip text for NaN cells
            # Calculate the intensity of the green color
            bg_color = cmap(harvest[i, j])
            intensity = bg_color[0]*299 + bg_color[1]*587 + bg_color[2]*114
            if intensity > 190:  # adjust this value to get the desired contrast
                text_color = 'black'
            else:
                text_color = 'white'
                
            text = ax.text(j, i, harvest[i, j], ha="center", va="center", color=text_color)

    if all_waves == True: text = f"{np.mean(avg_num_connections_per_ego_list, axis = 0).round(2)}"
    else:  text = f"{avg_num_connections_per_ego_list[w].round(2)}"
    ax.set_title(f"Correlations \n Avg tot connections: {text} \n")
    fig.tight_layout()
    plt.show()

#%%
