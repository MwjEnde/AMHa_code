

#%%
from code import interact
from functools import total_ordering
from genericpath import sameopenfile
from multiprocessing import get_all_start_methods
import numpy as np
import pandas as pd
import networkx as nx
import init_rcParams
init_rcParams.set_mpl_settings()
from statsmodels.stats import proportion
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import statsmodels.formula.api as smf

from statsmodels.stats import proportion
from statsmodels.tools import add_constant
from statsmodels.discrete import discrete_model
import statsmodels.api as sm
# import sys
# sys.path.append('/home/maarten/Documents/treasure')

with open('/home/maarten/Documents/treasure/data_files/main/df_list_nov.pkl', 'rb') as fp:
    df_list = pickle.load(fp)
    
dynamic_df = pd.read_csv('/home/maarten/Documents/treasure/data_files/main/df_nov.csv')


# %%
# df_list[0]
for i, df in enumerate(df_list):
    # Only take adults
    df = df.loc[df.age >= 21]
    # Only take where we know drinking data
    df = df.dropna(subset = ['d_state_ego','d_state_alter'], axis = 0)
    # Also drop where drinking state = -1 because we do not have sex
    #! doesnt work / doesnt solve it? check the -1
    df = df.loc[df.d_state_ego != -1]
    df_list[i] = df
#%%


# %%
ttnw = [9,5,4,4,4,3]
foo = pd.DataFrame(columns = ['wave', 'state', 'p stay in same state /year', 'p same state next exam'])
for state in range(0,3):
    ps_list = []
    for w in range(0,6):
        df_x = df_list[w][['shareid','d_state_ego']]
        df_y = df_list[w+1][['shareid','d_state_ego']]
        
        both= pd.merge(df_x, df_y, how = 'left', left_on='shareid', right_on='shareid' )
        both= both.dropna().astype(int)
        both = both.groupby('shareid').mean().reset_index().astype(int)

        # same = both.loc[((both.d_state_ego_x == both.d_state_ego_y) & (both.d_state_ego_x == state))]
        # prob_same = len(same)/len(both)

        diff = both.loc[((both.d_state_ego_x != both.d_state_ego_y) & (both.d_state_ego_x == state))]
        prob_diff = len(diff)/len(both.loc[both.d_state_ego_x == state])
        
        _dict = {'wave': w, 'state': state, 'p stay in same state /year': 1-prob_diff/ttnw[w], 'p same state next exam': 1- prob_diff}
        foo = pd.concat([foo, pd.DataFrame([_dict])], ignore_index = True)

k = foo.groupby(foo.state).mean().reset_index()
#%%
# CORRELATION TESTING
import pingouin as pg

corr_list = []
r_list = []
for w in range(0,6):
    df_x = df_list[w][['shareid','d_state_ego']]
    df_y = df_list[w+1][['shareid','d_state_ego']]
    
    both= pd.merge(df_x, df_y, how = 'left', left_on='shareid', right_on='shareid' )
    both= both.dropna().astype(int)
    both = both.groupby('shareid').mean().reset_index().astype(int)
    both = both.loc[(both.d_state_ego_x != -1) & (both.d_state_ego_y != -1)]

    plt.scatter(both.d_state_ego_x, both.d_state_ego_y)
    corr = pg.corr(both.d_state_ego_x, both.d_state_ego_y)

    corr_list.append(corr)
    r_list.append(corr.r.values)

print(np.asarray(r_list).mean())
print(corr_list)    

#%%
############### TRANSITION RATES PER WAVES INCLUDING TO WHO TABLE ##########

tot_np = np.empty((3,4))
for w in range(0,6):
    df_x = df_list[w][['shareid','d_state_ego']]
    df_y = df_list[w+1][['shareid','d_state_ego']]
    
    both= pd.merge(df_x, df_y, how = 'left', left_on='shareid', right_on='shareid' )
    both= both.dropna().astype(int)
    both = both.groupby('shareid').mean().reset_index().astype(int)
    both = both.loc[(both.d_state_ego_x != -1) & (both.d_state_ego_y != -1)]
    dummie = pd.get_dummies(both, columns = ['d_state_ego_y'])
    dummie = dummie.assign(tot_nb = 1)
    grouped = dummie.groupby(dummie.d_state_ego_x).sum().drop(columns = ['shareid']).reset_index().drop(columns = ['d_state_ego_x'])
    grouped_np = np.array(grouped.values)
    tot_np = tot_np + grouped_np
    
df = pd.DataFrame(tot_np)
# is from state at t=x to what state at t = x+1, total over all waves
perc = df.apply(lambda x: x/x.iloc[3], axis = 1).drop(columns = [3])

#%%
perc
#%%

# harvest =perc
# vegetables = ["abstaining", "moderate", "heavy"]
# farmers = ["abstaining", "moderate", "heavy"]
# fig, ax = plt.subplots()
# im = ax.imshow(harvest,cmap="Greens")

# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(farmers)), labels=farmers)
# ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, np.round(harvest.values[i, j],3),
#                     ha="center", va="center", color="#1C110A")

# ax.set_title(f"Probability\y to go to which state \n ")
# fig.tight_layout()
# plt.show()



#%%

################### SENSITIVITY ANALYSIS ON CUTOFF POINTS ####################
def get_stable_state_analysis(df_list, cutoff = 7):
    ttnw = [9,5,4,4,4,3]
    foo = pd.DataFrame(columns = ['wave', 'state', 
                     'p MALE stay in same state /year',
                     'p MALE same state next exam',
                     'p Female stay in same state /year',
                     'p Female same state next exam',
                     'p TOT stay in same state /year',
                     'p TOT same state next exam'
                     ])
    for state in range(0,3):
        for w in range(0,6):
            df_x = df_list[w][['shareid','dpw', 'sex_ego']]
            df_y = df_list[w+1][['shareid', 'dpw']]
            
            
            both = pd.merge(df_x, df_y, how = 'left', left_on='shareid', right_on='shareid' )
            both = both.dropna().astype(int)
            both = both.groupby('shareid').mean().reset_index().astype(int)

            def get_tot_stay_diff(df, state, cutoff):
                if state == 0:
                    diff = df.loc[(df.dpw_x == 0) & (df.dpw_y != 0)]
                    tot_in_state_x = df.loc[(df.dpw_x == 0)]
                if state == 1:                    
                    diff = df.loc[((df.dpw_x > 0) & (df.dpw_x <= cutoff)) & ((df.dpw_y == 0) | (df.dpw_y > cutoff))]
                    tot_in_state_x = df.loc[((df.dpw_x > 0) & (df.dpw_x <= cutoff))]
                if state == 2:
                    diff = df.loc[(df.dpw_x > cutoff) & (df.dpw_y <= cutoff)]
                    tot_in_state_x = df.loc[(df.dpw_x > cutoff)]
                
                return [diff, tot_in_state_x]
            
            # Get men stats
            men_df = both.loc[both.sex_ego == 1]
            cutoff_m = cutoff*2
            diff_m, tot_in_x_m = get_tot_stay_diff(men_df, state, cutoff_m)
            prob_diff_m = len(diff_m) / len(tot_in_x_m)
            
            # Get female stats
            fem_df = both.loc[both.sex_ego == 2]
            cutoff_f = cutoff
            diff_f, tot_in_x_f = get_tot_stay_diff(fem_df, state, cutoff_f)
            prob_diff_f = len(diff_f) / len(tot_in_x_f)
            
            # Tot stats:
            diff_tot_len = len(diff_m) + len(diff_f)
            tot_in_x_tot_len = len(tot_in_x_m) + len(tot_in_x_f)
            prob_diff_tot = diff_tot_len/tot_in_x_tot_len
            
            _dict = {'wave': w, 'state': state, 
                     'p MALE stay in same state /year': 1-prob_diff_m/ttnw[w], 
                     'p MALE same state next exam': 1- prob_diff_m,
                     'p Female stay in same state /year': 1-prob_diff_f/ttnw[w], 
                     'p Female same state next exam': 1- prob_diff_f,
                     'p TOT stay in same state /year': 1-prob_diff_tot/ttnw[w], 
                     'p TOT same state next exam': 1- prob_diff_tot,
                    }
            
            foo = pd.concat([foo, pd.DataFrame([_dict])], ignore_index = True)
    k = foo.groupby(foo.state).mean().reset_index()
    return k

kaas = get_stable_state_analysis(df_list)
foo = []
bound_list = range(1,20)
for i in bound_list:
    foo.append(get_stable_state_analysis(df_list, cutoff = i).values)
foo = np.asarray(foo)
    
#%%
# p tot stay per year plot ABSTAINING:
plt.plot(bound_list, foo[:,0,5], label="all")
plt.plot(bound_list, foo[:,0,3], label="female")
plt.plot(bound_list, foo[:,0,1], label="men")
plt.legend()
plt.xlabel('Moderate use upper bound (women, 2x for men)')
plt.ylabel('Stability /y ')
plt.title('Abstaining')
plt.axvline(x = 7, ls = ':')
plt.show()

#%%
# p tot stay per year plot MEDIUM DRINKING:
data = foo[:,1,5]
plt.plot(bound_list, data, label="all")
plt.plot(bound_list, foo[:,1,3], label="female")
plt.plot(bound_list, foo[:,1,1], label="men")
plt.legend()
plt.xlabel('Moderate use upper bound (women, 2x for men)')
plt.ylabel('Stability /y')
plt.title('Moderate drinking')
plt.axvline(x = 7, ls = ':')
plt.show()

#%%
# p tot stay per year plot Heavy:
plt.plot(bound_list, foo[:,2,5], label="all")
plt.plot(bound_list, foo[:,2,3], label="female")
plt.plot(bound_list, foo[:,2,1], label="men")
plt.legend()
plt.xlabel('Moderate use upper bound (women, 2x for men)')
plt.ylabel('Stability /y ')
plt.title('Heavy Drinking')
plt.axvline(x = 7, ls = ':')
plt.show()

#%%
# p tot stay per year plot all states all:
plt.plot(bound_list, foo[:,0,5], label="abstaining", linestyle = '-.')
plt.plot(bound_list, foo[:,1,5], label="moderate", linestyle = '--')
plt.plot(bound_list, foo[:,2,5], label="heavy", linestyle = ':')
plt.legend()
plt.xlabel('Moderate use upper bound (women, 2x for men)(/week)')
plt.ylabel('Stability /y ')
# plt.title('All states Drinking, men and women')
plt.axvline(x = 7, ls = '-')
plt.show()
#%%
# p tot stay per year plot all states MEN:
plt.plot(bound_list, foo[:,0,1], label="abstaining")
plt.plot(bound_list, foo[:,1,1], label="moderate")
plt.plot(bound_list, foo[:,2,1], label="heavy")
plt.legend()
plt.xlabel('Moderate use upper bound (women, 2x for men)(/week)')
plt.ylabel('Stability /y ')
plt.title('All states Drinking men')
plt.axvline(x = 7, ls = ':')
plt.show()
#%%
# p tot stay per year plot all states Women:
plt.plot(bound_list, foo[:,0,3], label="abstaining")
plt.plot(bound_list, foo[:,1,3], label="moderate")
plt.plot(bound_list, foo[:,2,3], label="heavy")
plt.legend()
plt.xlabel('Moderate use upper bound (women, 2x for men)(/week)')
plt.ylabel('Stability /y ')
plt.title('All states Drinking Women')
plt.axvline(x = 7, ls = ':')
plt.show()


#%%
# p tot stay per year plot medium added :

plt.plot(bound_list, np.asarray(foo[:,1,3]) + foo[:,2,3], label="combined Women")
plt.plot(bound_list, np.asarray(foo[:,1,1]) + foo[:,2,1], label="combined Men")

plt.legend()
plt.xlabel('Moderate use upper bound (women, 2x for men)(/week)')
plt.ylabel('Stability /y ')
plt.title('Sum of stability heavy + moderate')
plt.axvline(x = 7, ls = ':')
plt.show()


#%%
