
###### THIS CODE GENERATES Table 2, Figure 1, and Correlation Coefficient r in 'Stability of drinking states', and transition probabilities of section 'Three-state system'
#%%

import numpy as np
import pandas as pd
import init_rcParams
init_rcParams.set_mpl_settings()
import matplotlib.pyplot as plt
import pickle

#%%
with open('../../treasure/data_files/main/df_list_nov.pkl', 'rb') as fp:
    df_list = pickle.load(fp)

for i, df in enumerate(df_list):
    # Only take adults
    df = df.dropna(subset = ['d_state_ego','d_state_alter'], axis = 0)
    df = df.loc[df.d_state_ego != -1]
    df = df.loc[df.age >= 21]
    df_list[i] = df


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
# CORRELATION TESTIN
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

    # plt.scatter(both.d_state_ego_x, both.d_state_ego_y)
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
