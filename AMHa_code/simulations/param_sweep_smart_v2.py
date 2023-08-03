#%%
from init_rcParams import set_mpl_settings
set_mpl_settings()
#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing 
import statsmodels.stats.api as sms
#%%
import AMHA_class

#%%
def runProcess(param_name:str, param, number_of_iterations_sweep):
    model = AMHA_class.AmhaModel(**{param_name: param})
    output = model.simulate(custom_iterations=number_of_iterations_sweep)[:,-1]
    return output


def run_sweep(param_name, param_range, number_of_iterations_sweep):
    for i in range(len(param_name)):
        result_avg = np.zeros((n_sweeps,len(param_range), 3))
        for j in range(0, n_sweeps):
            multiProcPool = multiprocessing.Pool(processes=len(param_range))
            
            # Create tuples for i and param values
            args = [(param_name[i], param_range[j], number_of_iterations_sweep) for j in range(len(param_range))]
            # print(args)
            result = multiProcPool.starmap(runProcess, args) 
            
            #Save the result and close the pool
            result_avg[j] = result 
            multiProcPool.close()
            multiProcPool.join()
        with open(f'./pickles/sweep/{param_name[i]}_{param_range[0]}_{param_range[-1]}_{number_of_iterations_sweep}.pkl', 'wb') as f:
            pickle.dump(result_avg, f)
            
#%%    
# USE EITHER THIS        
#########! FLAT sweeps of social and automatic multipliers
param_name = ['flat_ai', 'flat_ar', 'flat_bi', 'flat_br']
param_range = np.arange(0.0, 0.201,0.01) 
n_sweeps = 33
number_of_iterations_sweep = 300
names = param_name
xlabels = ['Spontaneous Infection  $\u03B1$', 'Spontaneous Recovery  $\u03B1$', 'Social Infection  $\u03B2$', 'Social Recovery  $\u03B2$']
#%%
# OR THIS
########! MULTIPLIER sweeps of social and automatic multipliers
param_name = ['alpha_i_mp', 'alpha_r_mp', 'beta_i_mp', 'beta_r_mp']
param_range = np.arange(0.0, 2.01,0.01) 
n_sweeps = 33
number_of_iterations_sweep = 300
names = param_name
xlabels = ['Spontaneous Infection multiplier  $\u03B1$', 'Spontaneous Recovery multiplier  $\u03B1$', 'Social Infection multiplier $\u03B2$', 'Social Recovery multiplier  $\u03B2$']
#%%
### RUN SIMULATIONS & STORE SIMULATIONS IN PICKLE
# UNCOMMENT TO RUN -- TAKES LONG
# run_sweep(param_name, param_range, number_of_iterations_sweep)
# %%
### SUBPLOTS IN SQUARE
from matplotlib.ticker import ScalarFormatter

sqrt_len = int(np.ceil(np.sqrt(len(names))))
fig, axs = plt.subplots(nrows=sqrt_len, ncols=sqrt_len, figsize = (20,12))
for i in range(len(names)):
    row = i // sqrt_len
    col = i % sqrt_len
    with open(f'./pickles/sweep/{param_name[i]}_{param_range[0]}_{param_range[-1]}_{number_of_iterations_sweep}.pkl', 'rb') as fp:
        result_avg = pickle.load(fp)
    
    num_nodes =  result_avg.mean(axis = 0).sum(axis = 1)
    fractions_all_states = result_avg.mean(axis = 0) / num_nodes[:,None]
    lower_upper_bound = np.array(list(map(lambda x: list(sms.DescrStatsW(result_avg[:,x,2]).tconfint_mean()), range(len(param_range)))))/num_nodes[:,None]

    axs[row][col].set_xlabel(f'{xlabels[i]}')
    axs[row][col].set_ylabel('fraction heavy drinkers')
    #! LOG SCALE FOR MULTIPLIER WSS NIET HE
    # DONT USE THIS FOR FLAT SCALE
    axs[row][col].set_xscale('log')
    axs[row][col].xaxis.set_major_formatter(ScalarFormatter())
    axs[row][col].set_xlim([0.1, 2.1])
    axs[row][col].set_xticks([0.1, 0.5, 1, 2])
    # TILL HERE

    


    axs[row][col].plot(param_range, result_avg.mean(axis = 0)[:,2]/num_nodes)
    # axs[row][col].plot(fractions_all_states)
    axs[row][col].fill_between(param_range,lower_upper_bound[:,0],lower_upper_bound[:,1], alpha = 0.5)
    axs[row][col].set_xlim((param_range[0], param_range[-1]))
    # axs[row][col].savefig(f'./figures/generated/{names[i]}')

plt.show()

