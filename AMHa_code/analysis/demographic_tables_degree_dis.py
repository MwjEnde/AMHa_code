######## THIS CODE GENERATES Table 1, Supplementary Figure S3
######## Table 1 is an aggregation of the different tables compiled here


#%%
import numpy as np
import pandas as pd
import init_rcParams
init_rcParams.set_mpl_settings()
import matplotlib.pyplot as plt
import pickle


#%%
#%%
with open('../../treasure/data_files/main/df_list_nov.pkl', 'rb') as fp:
    df_list = pickle.load(fp)

for i, df in enumerate(df_list):
    # Only take adults
    df = df.dropna(subset = ['d_state_ego','d_state_alter'], axis = 0)
    df = df.loc[df.d_state_ego != -1]
    df = df.loc[df.age >= 21]
    df_list[i] = df


#%%
for count, wave in enumerate(df_list):

    wave = wave.assign(counter=1)
    contacts_total = len(wave)/wave.shareid.nunique()
    contacts_abstain = len(wave.loc[wave.d_state_alter.isin([0])])/wave.shareid.nunique()
    contacts_heavy_drink = len(wave.loc[wave.d_state_alter.isin([2])])/wave.shareid.nunique()


    row = {
        'wave, #' : count + 1,
        'Age': wave.groupby([wave.shareid])['age'].mean().mean(),
        'Egos total' : len(wave.groupby('shareid')),
        'Drinks per day': wave.groupby([wave.shareid])['dpw'].mean().mean() / 7,
        'Contacts total': contacts_total,
        'Contacts who abstain, n' : contacts_abstain,
        '% abstain' : round(contacts_abstain/contacts_total*100,0),
        'Contacts who drink heavily, n' : contacts_heavy_drink,
        '% heavy' : round(contacts_heavy_drink/contacts_total*100,0),
        
        
    }
    if count == 0:
        exp_df = pd.DataFrame(columns=list(row.keys()))

    exp_df.loc[count] = row

# calculate the total weight
total_weight = exp_df['Egos total'].sum()

# calculate the weighted mean for each column except 'wave, #' and 'Egos total'
for col in exp_df.columns:
    if col not in ['wave, #', 'Egos total']:
        exp_df.loc['all waves', col] = (exp_df[col] * exp_df['Egos total']).sum() / total_weight

exp_df = exp_df.round(2)
exp_df



#%%
# DEGREE DISTRIBUTION
# importing necessary modules for interactivity
from ipywidgets import interact, interactive, fixed, interact_manual,widgets
from matplotlib.ticker import PercentFormatter


@interact(w = (0,6))
def plot_degree_dist(w):
    #Choose the wave variable to be plotted
    global df
    global data
    df = df_list[w]
    # Assigning a counter variable and set it to 1
    df = df.assign(counter = 1)
    # Setting range for x-axis
    plt.xlim(0,15)
    # Setting labels for axes
    plt.ylabel('P(k)')
    plt.xlabel('Degree k')
    #Set x-axis ticks from 0 to 15
    plt.xticks(np.arange(0,15,1))
    # Calculate data for plotting 
    data = df.groupby(df.shareid).counter.sum().values
    # Plot Degree Distribution as histogram
    # y-axis will display percentage 
    plt.hist(data, weights=np.ones(len(data)) / len(data), bins = 15, range = (0,15), rwidth = 0.5, align = 'left')
    # Show mean value
    print(f"<k> = {data.mean():.2f}")
    plt.show()

#%%
# Degree Distribution for Heavy (2), Medium (1) and Abstain (0) Separated

from matplotlib.ticker import PercentFormatter

# initializing variables 
w = 2
df = df_list[w]
df = df.assign(counter=1)

# plotting the graph 
plt.xlim(0,10)
plt.ylabel('P(k)')
plt.xlabel('Degree k')
plt.xticks(np.arange(0, 10, 1))

# calculating frequencies of each type of variable
data_abst = df.loc[df.d_state_ego == 0].groupby(df.shareid).counter.sum().values
data_abst_w = np.ones(len(data_abst)) / len(data_abst)
data_med = df.loc[df.d_state_ego == 1].groupby(df.shareid).counter.sum().values
data_med_w = np.ones(len(data_med)) / len(data_med)
data_heavy = df.loc[df.d_state_ego == 2].groupby(df.shareid).counter.sum().values
data_heavy_w = np.ones(len(data_heavy)) / len(data_heavy)

# creating the corresponding data and weights arrays
data = [data_abst, data_med, data_heavy]
weights = [data_abst_w, data_med_w, data_heavy_w]

# creating the histogram with the correct styling and labels 
plt.hist(data, weights=weights, bins=10, range=(0,10), rwidth=0.8, align='left', 
         label=['abstain', 'moderate', 'heavy'], color=['#BFDBF7', '#A663CC', '#880D1E']) 

# getting the observed data for plotting and printing the mean
mean_k = df.groupby(df.shareid).counter.sum().values.mean()
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
print(f"<k> = {mean_k:.2f}")
plt.show()


# %%
