

#%%
from code import interact
from multiprocessing import get_all_start_methods
import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.stats import proportion
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from init_rcParams import set_mpl_settings
set_mpl_settings()
import statsmodels.formula.api as smf

from statsmodels.stats import proportion
from statsmodels.tools import add_constant
from statsmodels.discrete import discrete_model
import statsmodels.api as sm

#%%
# with open('../../treasure/data_files/main/df_list_nov.pkl', 'rb') as fp:
with open('../../treasure/data_files/main/df_list_aug.pkl', 'rb') as fp:
    df_list = pickle.load(fp)
    

# !! takes only adults
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

def get_stats(df_list, state_x, w, age_range = (0,0)):
    """this returns (per wave) for each shareid (grouped) with drinking state = state_x at wave 'w'  the number of:
    - total connections,
    - connections with drinking state 'state_alter'
    - the state of the ego at wave +1: state_ego_y
    
    If all_waves == True , then aggregates for all waves.

    Args:
        df_list (_type_): _description_
        state_x (_type_): _description_
        state_alter (_type_): _description_
        wave (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Only look at ego starting off with state x and alters with known d state
    df_x = df_list[w][['shareid','sharealterid','d_state_ego','d_state_alter','age']]
    df_x = df_x.loc[df_x.d_state_ego == state_x]
    df_x = df_x.loc[~df_x.d_state_alter.isna()]
    
    # Only look at ego within age range
    if age_range != (0,0):
        df_x = df_x.loc[np.logical_and(df_x.age >= age_range[0], df_x.age < age_range[1])]

    # Get d-state at wave n+1
    rel_y = df_list[w+1].loc[:,['shareid', 'd_state_ego']]
    rel_y['d_state_ego'] = rel_y.d_state_ego.fillna(-1).astype(int)
    rel_y = rel_y.groupby('shareid').mean().reset_index()
    rel_y['d_state_ego'] = rel_y['d_state_ego'].astype(int)
    df_x2 = df_x.merge(rel_y, left_on = 'shareid', right_on = 'shareid', how = 'inner').reset_index()
    # print(df_x2.info())

    # Drop rows where ego has missing data for drinking state at wave n+1 
    global df_x3
    # print(df_x3.columns)
    df_x3 = df_x2.loc[df_x2.d_state_ego_y != -1]
    df_x3 = df_x3.astype({'d_state_alter': int, 'd_state_ego_x': int})
    # df_x3['d_state_ego_y'] = df_x3.copy()['d_state_ego_y'].astype(int)
    # print(df_x3.info())
    # print(df_x3.columns)

    # Make it all dummied
    global lilia
    lilia = pd.get_dummies(df_x3, columns = ['d_state_alter'])
    lilia = lilia.assign(n_tot_nb = 1)
    lilia = pd.get_dummies(lilia, columns = ['d_state_ego_y'])

    # Make it all groupby shareid, by summing neighbors and mean (should all be the same) of ego data
    nb_data = lilia.loc[:,['shareid','d_state_alter_0','d_state_alter_1','d_state_alter_2','n_tot_nb']]
    nb_grouped = nb_data.groupby('shareid').sum().reset_index()
    ego_data = lilia.loc[:, ['shareid', 'd_state_ego_y_0',	'd_state_ego_y_1','d_state_ego_y_2', 'age']]
    ego_grouped = ego_data.groupby('shareid').mean().reset_index()

    # Merge neighbor and ego data to the same df
    stats = nb_grouped.merge(ego_grouped, left_on='shareid', right_on='shareid', how = 'left')

        
    return stats
#%%
def get_stats_all_waves(df_list, state_x, age_range = (0,0)):
    for w in range(0,6):   
        stats = get_stats(df_list, state_x, w, age_range)
        if w == 0: 
            tot_stats = stats
        else: tot_stats = pd.concat([tot_stats, stats], ignore_index= True)
    return tot_stats


#%%
############## PLOT STATS & LIN FIT PER WAVE 

from ipywidgets import interact, interactive, fixed, interact_manual,widgets
@interact(state_x = (0,2,1), state_y = (0,2,1), d_state_alter = (0,2,1), w = range(0,6), age_range_min= widgets.IntSlider(min = 0, max =100,step = 5, value =0), age_range_max = widgets.IntSlider(min = 30,max = 100, step = 5, value = 100), all_waves = False, grey_fit = False)
def plot_from_stats(state_x = 1, state_y = 2, d_state_alter = 2, w = 0, age_range_min = 0, age_range_max = 0, all_waves = False, grey_fit = False):
    
    
    age_range = (age_range_min, age_range_max)
    
    # global stats
    if all_waves: stats = get_stats_all_waves(df_list, state_x, age_range)
    else: stats = get_stats(df_list, state_x = state_x, w = w, age_range = age_range)

    global n_obs
    plot_stats = stats.groupby('d_state_alter_{}'.format(d_state_alter)).mean().reset_index()
    n_obs = stats.assign(num_ego = 1)[['d_state_alter_{}'.format(d_state_alter),'d_state_ego_y_{}'.format(state_y),'num_ego']].groupby('d_state_alter_{}'.format(d_state_alter)).sum().reset_index()
    
    # Plot data points 
    confidence_intervals = proportion.proportion_confint(n_obs.iloc[:,1],n_obs.iloc[:,2])
    y_err = np.array(confidence_intervals[1]) - plot_stats['d_state_ego_y_{}'.format(state_y)]
    
    markers, caps, bars = plt.errorbar(plot_stats.index,  plot_stats['d_state_ego_y_{}'.format(state_y)], yerr = y_err, fmt ='s' , ecolor = '#880D1E',  markersize = 10, color = '#1C110A', fillstyle = 'full')

    # if show_number: 
    for i in range(len(n_obs.index.values)):
        plt.annotate(str(n_obs.num_ego.values[i]), (plot_stats.index.values[i]+0.02, plot_stats['d_state_ego_y_{}'.format(state_y)].values[i]+0.04), fontweight = 'demi')
        

    # Linear Fit
    X = stats['d_state_alter_{}'.format(d_state_alter)].values
    X = add_constant(X)
    model = sm.OLS(endog = stats['d_state_ego_y_{}'.format(state_y)], exog = X).fit()

    # Plot fit
    if grey_fit == False:
        fit_color = '#C18203'
    else: fit_color = '0.8'
    
    if model.pvalues[1] <= 0.001:
        pval_text = ', p < 0.001'
    else: pval_text = f', p = {model.pvalues[1]:.3f}'
    if model.pvalues[0] <= 0.001:
        pval_text_alpha = ', p < 0.001'
    else: pval_text_alpha = f', p = {model.pvalues[0]:.3f}'
    a, b = model.params
    lb_a, lb_b = model.conf_int()[0]
    hb_a, hb_b = model.conf_int()[1]
    x_list = np.arange(0,7,1)
    line_low = lb_a + lb_b * x_list
    line_high = hb_a + hb_b * x_list
    plt.fill_between(x_list, y1 = line_low, y2 = line_high, alpha = 0.4, color = fit_color, 
                     label = f'$\u03B1$: {model.params[0]:.4f} {pval_text_alpha} \n$\u03B2$: {model.params[1]:.4f} {pval_text}')
    # plt.annotate('x1: {}, p: {}'.format(np.around(model.params[1],3),np.around(model.pvalues[1], 3)), xy = (0.1,0.1), xycoords='axes fraction')
    y = a + b * x_list
    plt.plot(x_list,y, '--', color = fit_color)
    plt.legend(loc='best')
    
    # Plot layout
    plt.xlim(-0.25,5.25)
    plt.ylim(0,0.45)
    state_name = {0: 'abstaining', 1: 'moderate', 2: 'heavy'}
    plt.ylabel('{} $\u2192$ {}'.format(state_name[state_x], state_name[state_y]))
    plt.xlabel("# of {} contacts".format(state_name[d_state_alter]))
    # plt.show()
    
#%%

#%%

plt.rcParams["figure.figsize"] = (20,14)
a = plt.figure()

ax1 = a.add_subplot(331)
# ax1.set_facecolor('#DBEBFB')
plot_from_stats(state_x = 0, state_y = 1, d_state_alter = 1, all_waves = True)
ax2 = a.add_subplot(332)
# ax2.set_facecolor('#DBEBFB')
plot_from_stats(state_x = 0, state_y = 1, d_state_alter = 2, all_waves = True)
ax3 = a.add_subplot(333)
# ax3.set_facecolor('#F6F0FA')
plot_from_stats(state_x = 0, state_y = 2, d_state_alter = 2, all_waves = True, grey_fit= True)
ax4 = a.add_subplot(334)
# ax4.set_facecolor('#DBEBFB')
plot_from_stats(state_x = 1, state_y = 0, d_state_alter = 0, all_waves = True)
# a.add_subplot(335)
ax5 = a.add_subplot(336)
# ax5.set_facecolor('#DBEBFB')
plot_from_stats(state_x = 1, state_y = 2, d_state_alter = 2, all_waves = True)
ax6 = a.add_subplot(337)
# ax6.set_facecolor('#F6F0FA')
plot_from_stats(state_x = 2, state_y = 1, d_state_alter = 1, all_waves = True, grey_fit= True)
ax7 = a.add_subplot(338)
# ax7.set_facecolor('#F6F0FA')
plot_from_stats(state_x = 2, state_y = 1, d_state_alter = 0, all_waves = True, grey_fit= True)
ax8 = a.add_subplot(339)
# ax8.set_facecolor('#DBEBFB')
plot_from_stats(state_x = 2, state_y = 0, d_state_alter = 0, all_waves = True)


plt.tight_layout()

plt.show()
plt.rcParams["figure.figsize"] = [12,8]

#%%


########################################################################################
###################### ALL FITS TABLE ONLY NB STATE ####################################
########################################################################################
########################################################################################

import statsmodels.formula.api as smf

from statsmodels.stats import proportion
from statsmodels.tools import add_constant
from statsmodels.discrete import discrete_model
import statsmodels.api as sm

fit_df = pd.DataFrame(columns = ['state_x', 'state_y', 'nb_state','x0','x0 p', 'x1','x1 p'])

for state_x in [0,1,2]:
    for state_y in [0,1,2]:
        for nb_state in [0,1,2]:

            for w in range(0,6):
                
                stats = get_stats(df_list, state_x = state_x, w = w, age_range = (0,0))
                stats = stats.assign(wave = w)
                
                if w == 0: 
                    tot_stats = stats
                else: tot_stats = pd.concat([tot_stats, stats], ignore_index= True)
                
                

            stats = tot_stats
            X = stats['d_state_alter_{}'.format(nb_state)].values

            X = add_constant(X)
            model = sm.OLS(endog = stats['d_state_ego_y_{}'.format(state_y)], exog = X).fit()
            x_list = np.arange(0,8,1)
            a, b = model.params

            fit_dict = {
                'state_x': state_x,
                'state_y': state_y,
                'nb_state': nb_state,
                'x0' : a,
                'x0 p' : model.pvalues[0],
                'x1' : b,
                'x1 p' : model.pvalues[1],
                'lower_conf_x0': model.conf_int()[0][0],
                'lower_conf_x1': model.conf_int()[0][1],
                'higher_conf_x0': model.conf_int()[1][0],
                'higher_conf_x1': model.conf_int()[1][1]
            }
            fit_df = pd.concat([fit_df, pd.DataFrame([fit_dict])], ignore_index = True)
fit_df = fit_df.round(3)

# %%
fit_df
# %%

# %%
#################### FIT PER WAVE ###############################################################
fit_df_list = []
for w in range(0,6):
    fit_df = pd.DataFrame(columns = ['state_x', 'state_y', 'nb_state','x0','x0 p', 'x1','x1 p', 'num_obs'])
    for state_x in [0,1,2]:
        for state_y in [0,1,2]:
            for nb_state in [0,1,2]:


                # global stats
                stats = get_stats(df_list, state_x = state_x, w = w, age_range = (0,0))
                stats = stats.assign(wave = w)
                

                # num_observations = stats.loc[(stats[f'd_state_ego_y_{state_y}'] == 1)][f'd_state_alter_{nb_state}'].sum()
                global n_obs
                n_obs_df = stats.assign(num_ego = 1)[['d_state_alter_{}'.format(nb_state),'d_state_ego_y_{}'.format(state_y),'num_ego']].groupby('d_state_alter_{}'.format(nb_state)).sum().reset_index()
                n_obs = n_obs_df.num_ego.sum()


            # stats = tot_stats
                X = stats['d_state_alter_{}'.format(nb_state)].values
                X = add_constant(X)
                model = sm.OLS(endog = stats['d_state_ego_y_{}'.format(state_y)], exog = X).fit()
                x_list = np.arange(0,8,1)
                a, b = model.params

                fit_dict = {
                    'state_x': state_x,
                    'state_y': state_y,
                    'nb_state': nb_state,
                    'x0' : a,
                    'x0 p' : model.pvalues[0],
                    'x1' : b,
                    'x1 p' : model.pvalues[1],
                    'num_obs' : n_obs,
                    'lower_conf_x0': model.conf_int()[0][0],
                    'lower_conf_x1': model.conf_int()[0][1],
                    'higher_conf_x0': model.conf_int()[1][0],
                    'higher_conf_x1': model.conf_int()[1][1]
                }
                fit_df = pd.concat([fit_df, pd.DataFrame([fit_dict])], ignore_index = True)
                # print(fit_df)
                
    fit_df = fit_df.round(3)  
    fit_df_list.append(fit_df)
#%%
fit_df_list          

# %%

################ RATES OVER WAVES AUTOMATIC #####################
global plot_counter
plot_counter = 0
def plot_rates_per_wave(state_x, state_y, nb_state):

# state_x = 0
# state_y = 1
# nb_state = 1

    for i, df in enumerate(fit_df_list):
        global inf_param
        global relevant_params
        relevant_params = df.loc[((df.state_x == state_x) & (df.state_y == state_y)) & (df.nb_state == nb_state)].copy()

        if i == 0:
            inf_param = relevant_params

        
        else: 
            inf_param = pd.concat([inf_param, relevant_params], ignore_index = True)
            # rec_param = pd.concat([rec_param, auto_recovery], ignore_index=True)

    plt.errorbar(inf_param.index+1,  inf_param.x0, yerr = inf_param['x0 p'], fmt ='d' , ecolor = '#A663CC',  markersize = 15, color = '#A663CC', fillstyle = 'full', capsize = 5, label = r'automatic infection rate $\alpha$')
    # plt.errorbar(rec_param.index+1,  rec_param.x0, yerr = rec_param['x0 p'], fmt ='d' , ecolor = '#880D1E',  markersize = 15, color = '#880D1E', fillstyle = 'full', capsize = 5, label = r'recovery rate $g$')
        
    for i in range(6):

            
        plt.annotate(str(inf_param.num_obs.values[i]), (i+1+0.1, inf_param.x0.values[i]-0.01), fontweight = 'demi',color = '#A663CC')

        
    plt.xlabel('exam')
    plt.ylabel(r'rate (/wave)')
    plt.ylim(bottom = -.05)
    plt.ylim(top = 0.40)
    # plt.xlim(left = -0.05)
    state_name = {0: 'abstaining', 1: 'moderate', 2: 'heavy'}
    plot_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}
    plt.title(f'({plot_letter[plot_counter]}) {state_name[state_x]} $\u2192$  {state_name[state_y]} | {state_name[nb_state]} nb ')#, fontname="Times New Roman")
    if plot_counter==0:
        plt.legend()

a = plt.figure()

a.add_subplot(321)
plot_rates_per_wave(state_x = 0, state_y = 1, nb_state = 1)
plot_counter += 1

a.add_subplot(322)
plot_rates_per_wave(state_x = 0, state_y = 2, nb_state = 2)
plot_counter += 1

a.add_subplot(323)
plot_rates_per_wave(state_x = 1, state_y = 0, nb_state = 0)
plot_counter += 1

a.add_subplot(324)
plot_rates_per_wave(state_x = 1, state_y = 2, nb_state = 2)
plot_counter += 1

a.add_subplot(325)
plot_rates_per_wave(state_x = 2, state_y = 0, nb_state = 0)
plot_counter += 1

a.add_subplot(326)
plot_rates_per_wave(state_x = 2, state_y = 1, nb_state = 1)
plot_counter += 1

plt.rcParams["figure.figsize"] = (20,15)
plt.tight_layout()
plt.show()
plot_counter = 0



#%%


################ RATES OVER WAVES SOCIAL #####################
# global plot_counter
plot_counter = 0
def plot_rates_per_wave(state_x, state_y, nb_state):

# state_x = 0
# state_y = 1
# nb_state = 1

    for i, df in enumerate(fit_df_list):
        relevant_params = df.loc[((df.state_x == state_x) & (df.state_y == state_y)) & (df.nb_state == nb_state)].copy()
        
        
        if i == 0:
            inf_param =  relevant_params

            
        else: 
            inf_param = pd.concat([inf_param, relevant_params], ignore_index = True)

    plt.errorbar(inf_param.index+1,  inf_param.x1, yerr = inf_param['x1 p'], fmt ='d' , ecolor = '#880D1E',  markersize = 15, color = '#880D1E', fillstyle = 'full', capsize = 5, label = r'social infection rate $\beta$')
    
    for i in range(6):     
        plt.annotate(str(inf_param.num_obs.values[i]), (i+1+0.1, inf_param.x1.values[i]+0.01), fontweight = 'demi',color = '#880D1E')




    plt.xlabel('exam')
    plt.ylabel(r'rate (/wave)')
    plt.ylim(bottom = -.02)
    plt.ylim(top = 0.10)
    state_name = {0: 'abstaining', 1: 'moderate', 2: 'heavy'}
    plot_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g'}
    plt.title(f'({plot_letter[plot_counter]}) {state_name[state_x]} $\u2192$  {state_name[state_y]} | {state_name[nb_state]} nb ')#, fontname="Times New Roman")
    if plot_counter == 0:
        plt.legend()

a = plt.figure()


a.add_subplot(321)
plot_rates_per_wave(state_x = 0, state_y = 1, nb_state = 1)
plot_counter += 1

a.add_subplot(322)
plot_rates_per_wave(state_x = 0, state_y = 2, nb_state = 2)
plot_counter += 1

a.add_subplot(323)
plot_rates_per_wave(state_x = 1, state_y = 0, nb_state = 0)
plot_counter += 1

a.add_subplot(324)
plot_rates_per_wave(state_x = 1, state_y = 2, nb_state = 2)
plot_counter += 1

a.add_subplot(325)
plot_rates_per_wave(state_x = 2, state_y = 0, nb_state = 0)
plot_counter += 1

a.add_subplot(326)
plot_rates_per_wave(state_x = 2, state_y = 1, nb_state = 1)
plot_counter += 1

plt.rcParams["figure.figsize"] = (20,15)
plt.tight_layout()
plt.show()
plot_counter = 0



#%%
# 

