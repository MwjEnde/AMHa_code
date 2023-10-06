#%%

from init_rcParams import set_mpl_settings
set_mpl_settings()
#%%
import numpy as np
import matplotlib.pyplot as plt
import network_from_pd as FHS_net
import pandas as pd
import pickle
import multiprocessing 
import statsmodels.stats.weightstats as sms
#%%
import AMHA_class
amha_class = AMHA_class.AmhaModel(FHS_net.G_relabled)

#%%
number_of_iterations_sweep = 400
n_sweeps = 10
#%%

#%%
import multiprocessing
# import numpy as np
# from scipy import stats

def run_model(iterations):
    model = AMHA_class.AmhaModel(FHS_net.G_relabled, number_of_iterations=iterations)
    results = model.simulate()
    return results[0], results[1], results[2]

# Set the number of sweeps (n) and the number of processes to use
num_processes = multiprocessing.cpu_count()

# Create a Pool of worker processes
pool = multiprocessing.Pool(processes=num_processes)

# Run the model simulations in parallel
results = np.asarray(pool.map(run_model, [number_of_iterations_sweep] * n_sweeps))
pool.close()
pool.join()

#%%
### OBTAIN DATA POINTS
with open('../../treasure/data_files/main/df_list_nov.pkl', 'rb') as fp:
    df_list = pickle.load(fp) 
    
for i, df in enumerate(df_list):
    # Only take where we know drinking data
    df = df.dropna(subset = ['d_state_ego','d_state_alter'], axis = 0)
    # Also drop where drinking state = -1 because we do not have sex
    df = df.loc[df.d_state_ego != -1]
    # Only take adults
    df = df.loc[df.age >= 21]
    df_list[i] = df

prevalence_df_list = []
#%%
# Loop through each dataframe in df_list
for df in df_list:

    # Calculate the counts for each unique value of 'd_state_ego'
    counts = df['d_state_ego'].value_counts()
    # Divide each count by the total number of rows to get the relative prevalence
    prevalence = counts / len(df)
    # Add the results to a new dataframe
    prevalence_df = pd.DataFrame({'State': prevalence.index, 'Prevalence': prevalence.values})
    # Append the dataframe to a list
    prevalence_df_list.append(prevalence_df)
    
# Combine all the dataframes in the list into one
all_prevalences = pd.concat(prevalence_df_list)

#%%
#### OBSERVED PREVALENCES AND FUTURE PREDICTIONS

import pandas as pd
import matplotlib.pyplot as plt

n_iter = number_of_iterations_sweep+1 # For 50 iterations per year


# Load prevalence_data.csv into a dataframe
# prevalence_df = pd.read_csv('prevalence_data_5.csv')
prevalence_df = all_prevalences

# Define the x-axis locations for the data points
dates = [1972,1981, 1985, 1989, 1993, 1997, 2000]

    
num_nodes = results.sum(axis=1)[0]
result_avg_frac = results / num_nodes 
result_avg = result_avg_frac


labels = {0: 'Abstain', 1:'Moderate', 2: 'Heavy'}
plt.gca().set_prop_cycle(None)
colors = ['#BFDBF7', '#A663CC', '#880D1E']
markers = ['d','o', '+']

# Extract the relevant data from prevalence_df and create a scatter plot
for state in range(3):
# for state in [0,3]:
    y_values = prevalence_df[prevalence_df['State']==state]['Prevalence']
    plt.scatter(dates, y_values, marker=markers[state], alpha=0.99, label = labels[state],color=colors[state], s=100)
    
for state in range(3):
    mean = result_avg[:,state,:].mean(axis=0)
    plt.plot(np.linspace(1985,(n_iter/10)+1985,n_iter), mean, label=labels[state], color=colors[state])


for state in [0,2]:
    lower_upper_bound = np.array(list(map(lambda x: list(sms.DescrStatsW(result_avg[:,state,x]).tconfint_mean()), range(n_iter))))
    plt.fill_between(np.linspace(1985,(n_iter/10)+1985,n_iter),lower_upper_bound[:,0],lower_upper_bound[:,1], alpha=0.5, color=colors[state])

# Add axis labels and legend
plt.ylabel('Fraction in drinking state')
plt.xlabel('Years')
plt.ylim((0.0,0.6))
plt.xlim(right = 2025)
plt.legend(loc = 'lower right')

# Show the plot
plt.show()




# %%
