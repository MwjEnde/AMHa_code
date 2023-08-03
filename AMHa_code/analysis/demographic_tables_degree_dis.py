#%%

import numpy as np
import pandas as pd
import networkx as nx
import init_rcParams
from statsmodels.stats import proportion

init_rcParams.set_mpl_settings()
import matplotlib.pyplot as plt

import pickle

from pathlib import Path

#%%
    
with open('./data_files/main/df_list_aug.pkl', 'rb') as fp:
    df_list = pickle.load(fp)

# Load another dataframe from a CSV file
dynamic_df = pd.read_csv('./data_files/main/df_new.csv')

# Filter the resulting list of dataframes to only include
# adults and those whose alcohol use is known
# for i, df in enumerate(df_list):
#     # Only take adults
#     df_list[i] = df.loc[df.age >= 21]
#     # Only take entries when alcohol use is known
#     df_list[i] = df.dropna(subset = ['d_state_ego','d_state_alter'], axis = 0)


#%%
############################### Statistics #############################
for count, wave in enumerate(df_list):

    
    row = {
        # '# ego': wave.shareid.nunique(),
        # '# ego og':  wave.loc[wave.idtype == 0].shareid.nunique(),
        # '# ego of':  wave.loc[wave.idtype == 1].shareid.nunique(),
        # '# alter': wave.sharealterid.nunique(),
        'wave, #' : count + 1,
        '# ego OG with drinking data' : wave.loc[np.logical_and(wave.idtype == 0, ~wave.dpw.isnull())].shareid.nunique(),
        '# ego OF with drinking data' : wave.loc[np.logical_and(wave.idtype == 1, ~wave.dpw.isnull())].shareid.nunique(),
        '# ego tot with drinking data' : wave.loc[~wave.dpw.isnull()].shareid.nunique(),
        'dpw mean_tot':
        wave.groupby([wave.shareid])['dpw'].mean().mean() / 7,
        'dpw mean_original':
        wave.loc[wave.idtype == 0].groupby([wave.shareid
                                            ])['dpw'].mean().mean() / 7,
        'dpw mean_offspring':
        wave.loc[wave.idtype == 1].groupby([wave.shareid
                                            ])['dpw'].mean().mean() / 7,
        'avg age':
        wave.groupby([wave.shareid])['age'].mean().mean(),
    }

    if count == 0:
        exp_df = pd.DataFrame(columns=list(row.keys()))

    exp_df.loc[count] = row
    exp_df.set_index('wave, #')
    exp_df.loc['all waves'] = exp_df.mean(axis = 0)

exp_df = exp_df.round(2)
#%%
############################### Statistics #############################
# loop over the items in df_list
for count, wave in enumerate(df_list):

    # add a 'counter' column to the wave data
    wave = wave.assign(counter=1)

    # create the row dictionary for this wave
    row = {
        # Calculate number of unique participant ids for wave
        'wave, #' : count + 1,
        'Contacts, n' : wave.loc[wave.idtype == 1].groupby('shareid').counter.sum().mean(),
        'Close friends, n' : wave.loc[wave.ALTERTYPE == 'FRIENDNR'].groupby('shareid').counter.sum().mean(),
        'Close friends offspring, n' : wave.loc[np.logical_and(wave.ALTERTYPE == 'FRIENDNR',wave.idtype == 1)].groupby('shareid').counter.sum().mean(),
        'Close friends og, n' : wave.loc[np.logical_and(wave.ALTERTYPE == 'FRIENDNR',wave.idtype == 0)].groupby('shareid').counter.sum().mean(),
        'Family members, n' : wave.loc[wave.ALTERTYPE.isin(['CHILD','SISTER','BROTHER','MOTHER','FATHER']) & wave.idtype == 1].groupby('shareid').counter.sum().mean(),
        'sisbro': wave.loc[(wave.ALTERTYPE.isin(['SISTER','BROTHER']) & np.logical_and(wave.idtype == 1, wave.alteridtype == 1))].groupby('shareid').counter.sum().sum(),
        'sisbro2': wave.loc[wave.ALTERTYPE.isin(['SISTER','BROTHER'])].groupby('shareid').counter.sum().sum(),
    }

    # create the empty dataframe on the first loop run
    if count == 0:
        social_stats = pd.DataFrame(columns=list(row.keys()))

    # add the row data to the dataframe
    social_stats.loc[count] = row
    # set the index of the dataframe to 'wave, #'
    social_stats.set_index('wave, #')
    # calculate the mean of each column for all waves and add that row to dataframe
    social_stats.loc['all waves'] = social_stats.mean(axis = 0)
    # round the numbers in the dataframe to 2 decimal points
    social_stats = social_stats.round(2)
    
# display the compiled data frame
social_stats
# %%


exp_df
# %%

# Iterate through df_list and iterate with counter
for count, wave in enumerate(df_list):
    # Assign 'counter' column to each wave
    wave = wave.assign(counter=1)

    # Define variables to store information
    row = {
        'wave, #' : count + 1,
        'Contacts, n' : wave.loc[wave.idtype == 1].groupby('shareid').counter.sum().mean(),
        'Close friends, n' : wave.loc[wave.ALTERTYPE == 'FRIENDNR'].groupby('shareid').counter.sum().mean(),
        'Close friends offspring, n' : wave.loc[np.logical_and(wave.ALTERTYPE == 'FRIENDNR',wave.idtype == 1)].groupby('shareid').counter.sum().mean(),
        'Close friends og, n' : wave.loc[np.logical_and(wave.ALTERTYPE == 'FRIENDNR',wave.idtype == 0)].groupby('shareid').counter.sum().mean(),
        'Family members, n' : wave.loc[wave.ALTERTYPE.isin(['CHILD','SISTER','BROTHER','MOTHER','FATHER']) & wave.idtype == 1].groupby('shareid').counter.sum().mean(),
        'sisbro': wave.loc[(wave.ALTERTYPE.isin(['SISTER','BROTHER']) & np.logical_and(wave.idtype == 1, wave.alteridtype == 1))].groupby('shareid').counter.sum().sum(),
        'sisbro2': wave.loc[wave.ALTERTYPE.isin(['SISTER','BROTHER'])].groupby('shareid').counter.sum().sum(),
    }

    # Create a dataframe if processing first wave
    if count == 0:
        social_stats = pd.DataFrame(columns=list(row.keys()))

    # Append information for each wave to the dataframe
    social_stats.loc[count] = row
    social_stats.set_index('wave, #')
    social_stats.loc['all waves'] = social_stats.mean(axis = 0)

    # Round values in the dataframe for better readability
    social_stats = social_stats.round(2)

    # Show the dataframe
social_stats


#%%
################################ SOCIAL NET EXPLORE TABLE ###################################
# Loop over enumerated df_list
for count, wave in enumerate(df_list): 
    # Create column counter and set value to 1
    wave = wave.assign(counter=1)
    
    # Keep only directed FRIENDNR, SAMEADNREL, SPOUSE where age >= 21
    wave = wave.loc[wave.age >= 21]
    wave = wave.loc[wave.idtype == 1]
   
    # Create dict of row with relevant info
    row = {
        'wave, #' : count + 1,
        
        # Calculate Total contacts
        'Total contacts, n' : len(wave)/wave.shareid.nunique(),
        
        # Count total contacts with drinking data
        'Total contacts with drinking data, n' : len(wave.loc[~wave.dpw_alter.isna()])/wave.shareid.nunique(),
        
        # Count Close FRIENDNR (excluding SameAdnRel and Spouse)
        'Close FRIENDNR, n' : len(wave.loc[wave.ALTERTYPE.isin(['FRIENDNR' ])])/wave.shareid.nunique(),
        
        # Count Close friends wide (including SameAdnRel and Spouse)
        'Close friends wide, n' : len(wave.loc[wave.ALTERTYPE.isin(['FRIENDNR','SAMEADNREL', ])])/wave.shareid.nunique(),
        
        # Count Close friends wide directional (excluding reciprocal friendships)
        'Close friends wide directional, n' : len(wave.loc[
            np.logical_and(
                wave.ALTERTYPE.isin(['FRIENDNR','SPOUSE', 'SAMEADNREL']), 
                wave.CAUSEINIT != 'reciprocal_friend'
                )
            ])
        /wave.shareid.nunique(),
        
        # Count Family members
        'Family members, n' : len(wave.loc[wave.ALTERTYPE.isin(['CHILD','SISTER','BROTHER','MOTHER','FATHER','SPOUSE','RELATIVENR'])])/wave.shareid.nunique(),
        
        # Count Adult Family members (age >= 21)
        'Adult family members, n' :  len(wave.loc[np.logical_and(wave.ALTERTYPE.isin(['CHILD','SISTER','BROTHER','MOTHER','FATHER','SPOUSE','RELATIVENR']), np.logical_or(wave.alter_age >= 21, wave.alter_age.isna()))])/wave.shareid.nunique(),
        
        # Count contacts who abstain
        'Contacts who abstain, n' : len(wave.loc[wave.d_state_alter.isin([0])])/wave.shareid.nunique(),
        
        # Count contacts who drink heavily
        'Contacts who drink heavily, n' : len(wave.loc[wave.d_state_alter.isin([2])])/wave.shareid.nunique()
    
    }

    # Check if count is 0
    if count == 0:
        # Create empty table_1 df and columns from row
        table_1 = pd.DataFrame(columns=list(row.keys()))

    # Add row of info to table
    table_1.loc[count] = row
    # Set wave to be index
    table_1.set_index('wave, #')
    # Add average of table to table
    table_1.loc['all waves'] = table_1.mean(axis = 0)

    # Organize table and round to 2 decimal places
    table_1 = table_1.round(2)
    # table_1 = table_1.set_index('wave, #')

    # Print table
table_1


# Cleaned-up and Commented Code 
#%%
#* Drinking patterns over age MALE
# Iterate through dataframe list and select only the rows from 'sex_ego' that have value of 1
for i, df in enumerate(df_list):
    df = df.loc[df.sex_ego == 1]
    # Plot Average Drinking Patterns for male over age 
    # label argument is to assign wave number for each plot line
    plt.plot(df.groupby('age').dpw.mean(), label = 'wave {}'.format(i+1))
    plt.legend()
    plt.plot()
#%%
#* Drinking patterns over age FEMALE
# Iterate through dataframe list and select only those rows that have value of 2 in 'sex_ego'
for i, df in enumerate(df_list):
    df = df.loc[df.sex_ego == 2]
    # Plot Average Drinking Patterns for female over age
    # label argument is to assign wave number for each plot line
    plt.plot(df.groupby('age').dpw.mean(), label = 'wave {}'.format(i+1))
    plt.legend()
    plt.plot()

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
    data = df.groupby(dynamic_df.shareid).counter.sum().values
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
w = 5 
df = df_list[w]
df = df.assign(counter=1)

# plotting the graph 
plt.xlim(0,10)
plt.ylabel('P(k)')
plt.xlabel('Degree k')
plt.xticks(np.arange(0, 10, 1))

# calculating frequencies of each type of variable
data_abst = df.loc[df.d_state_ego == 0].groupby(dynamic_df.shareid).counter.sum().values
data_abst_w = np.ones(len(data_abst)) / len(data_abst)
data_med = df.loc[df.d_state_ego == 1].groupby(dynamic_df.shareid).counter.sum().values
data_med_w = np.ones(len(data_med)) / len(data_med)
data_heavy = df.loc[df.d_state_ego == 2].groupby(dynamic_df.shareid).counter.sum().values
data_heavy_w = np.ones(len(data_heavy)) / len(data_heavy)

# creating the corresponding data and weights arrays
data = [data_abst, data_med, data_heavy]
weights = [data_abst_w, data_med_w, data_heavy_w]

# creating the histogram with the correct styling and labels 
plt.hist(data, weights=weights, bins=10, range=(0,10), rwidth=0.8, align='left', 
         label=['abstain', 'moderate', 'heavy'], color=['#BFDBF7', '#A663CC', '#880D1E']) 

# getting the observed data for plotting and printing the mean
data = df.groupby(dynamic_df.shareid).counter.sum().values
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
print(f"<k> = {data.mean():.2f}")
plt.show()

# %%
# for df in df_list:
df = df_list[2]
df_group =  df.groupby(df.shareid).d_state_ego.mean().reset_index()
df_dummie = pd.get_dummies(df_group.d_state_ego)
df_dummie.value_counts().values[:-1]
# %%
number_of_people_in_each_state_per_wave = np.zeros((7,3))
fraction_of_people_in_each_state_per_wave = np.zeros((7,3))

for i, df in enumerate(df_list):
    df_group =  df.groupby(df.sharealterid).d_state_alter.mean().reset_index()
    df_dummie = pd.get_dummies(df_group.d_state_alter)
    number_of_people_in_each_state_per_wave[i] = df_dummie.value_counts().values[:-1]
    fraction_of_people_in_each_state_per_wave[i] = df_dummie.value_counts().values[:-1]/len(df_group)
# %%

# %%
