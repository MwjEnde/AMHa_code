#%%
# import init_rcParams
# init_rcParams.set_mpl_settings()
#%%
import pandas as pd
import numpy as np

import import_clinical_datasets

from pathlib import Path
data_folder = '../'/Path.home()/'Documents/HPC_treasure/hpc/sratoolkit.2.11.1-ubuntu64/bin/dbGaP-27276/'

#%%
def extract_social_net(include = None, exclude = ['N100MNREL'], singular = None):
    """Takes the social network downloaded data and turns it into dataframe. By default only removes N100MNREL, keeps neighbours 25 meters away. If wanted a single type of connection, use singular. Removes Distance and Treimann columns
    
    Args:
        exclude (list, optional): list of social connections to exclude. Defaults to ['N100MNREL']. Note that if user passes its own, then N100MNREL will be included again by default. If
    
    singular (string, optional): if given, will change from exclude to include a single type of connections; return connections of this type, e.g. 'FRIENDNR'. Defaults to None.

    Returns:
        DataFrame: social network of specified types of connection
    """    
    MDS_df = pd.read_csv(data_folder/"social_net/phs000153.v9.pht000836.v8.p8.c1.vr_sntwk_2008_m_0641s.DS-SN-IRB-MDS.txt", skiprows=10, delimiter='\t')
    NPU_df = pd.read_csv(data_folder/"social_net/phs000153.v9.pht000836.v8.p8.c2.vr_sntwk_2008_m_0641s.DS-SN-IRB-NPU-MDS.txt", skiprows=10, delimiter='\t')
    social_net = MDS_df.append(NPU_df)
    
    if singular != None:
        social_net = social_net[social_net.ALTERTYPE == singular]
    elif include != None:
        social_net = social_net.loc[social_net.ALTERTYPE.isin(include)]
    else:
        for exclude_column in exclude:
            social_net = social_net[social_net.ALTERTYPE != exclude_column]
    
    social_net = social_net.filter(regex = '^(?!(DIST.*))')
    # social_net = social_net.filter(regex = '^(?!(.*TREIMAN.*))')
    social_net.drop('dbGaP_Subject_ID', axis= 1, inplace= True)
    
    return social_net

#%%
def reciprocate_friends(sn):
    
    sn_friends_only = sn.loc[np.logical_and(sn.ALTERTYPE == 'FRIENDNR', sn.alteridtype == (0|1))]
    
    
    sn_friends_switched = sn_friends_only.rename(columns = {
        'EGO_TREIMAN1' : 'ALTER_TREIMAN1', 'EGO_TREIMAN2' : 'ALTER_TREIMAN2', 'EGO_TREIMAN3': 'ALTER_TREIMAN3', 'EGO_TREIMAN4': 'ALTER_TREIMAN4', 'EGO_TREIMAN5': 'ALTER_TREIMAN5',    'EGO_TREIMAN6': 'ALTER_TREIMAN6', 'EGO_TREIMAN7': 'ALTER_TREIMAN7', 'EGO_TREIMAN8': 'ALTER_TREIMAN8', 'ALTER_TREIMAN1' : 'EGO_TREIMAN1', 'ALTER_TREIMAN2' : 'EGO_TREIMAN2', 'ALTER_TREIMAN3' : 'EGO_TREIMAN3', 'ALTER_TREIMAN4' : 'EGO_TREIMAN4', 'ALTER_TREIMAN5' :'EGO_TREIMAN5','ALTER_TREIMAN6' : 'EGO_TREIMAN6', 'ALTER_TREIMAN7' : 'EGO_TREIMAN7','ALTER_TREIMAN8' : 'EGO_TREIMAN8','idtype':'alteridtype', 'alteridtype':'idtype','shareid':'sharealterid', 'sharealterid':'shareid'
        })

    sn_friends_switched = sn_friends_switched.assign(CAUSEINIT = 'reciprocal_friend')
    sn_friends_switched = sn_friends_switched.assign(CAUSESEVERED = 'reciprocal_friend')

    both_sn = pd.concat([sn, sn_friends_switched],ignore_index = True)
    return both_sn

#%%
def add_lapse(social_net):
    # This functions obtainse the time in months after an arbitrary date for the first exam to take place and adds it to the dataset.
    lapse_c1 = pd.read_csv(data_folder/'social_net/phs000153.v9.pht000837.v6.p8.c1.vr_lapse_2008_m_0649sn.DS-SN-IRB-MDS.txt', skiprows = 10, delimiter = '\t')
    lapse_c2 = pd.read_csv(data_folder/'social_net/phs000153.v9.pht000837.v6.p8.c2.vr_lapse_2008_m_0649sn.DS-SN-IRB-NPU-MDS.txt', skiprows = 10, delimiter = '\t')

    lapse = lapse_c1.append(lapse_c2).set_index('shareid')
    lapse = lapse.rename(columns = {'LAPSE':'date1'}).date1

    new_soc_net = social_net.merge(lapse, left_on = 'shareid', right_index = True, how = 'left')
    return new_soc_net


#%%
def import_dpw_cpw_age(s_net):    
    """imports the drinks per week from 'import__clinical_datasets' script, and merges it with the social network data. Further, it adds the data from the 'meta' file, containing CPD: cigarettes and Age per day. (only for offspring)
    
    closest available moment for each offspring wave exam date. This corresponds to exam 12, 17, 21, 22, 23 -> in my own calculation:
    # # def original_to_7_waves():
    #     midpoints: [1972/3,1981,1989/90, 1993,1996/7, 1999/2000]
    #     middle:    12,      17,   21 ,  22,  23,  24, 26
    #     # closest to middle point, with alcohol data available
    # 
    
    But-> data says The time interval includes Original Cohort exams 12, 16, 19, 21, 23, 24, 26, 29 (https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/variable.cgi?study_id=phs000153.v9.p8&phv=73871)
    #! 16 has no alcohol data so we take 17
    # 
    
    Returns:
        DataFrame: social_net dataframe, with added columns total_x dpw for both original and offspring cohorts
    """   
    ########################## Add offspring DPW ###########################
    drinks_per_week = import_clinical_datasets.import_clinical_datasets_drinks_PW()
    dynamic_df = s_net.merge(drinks_per_week, left_on = 'shareid', right_index = True, how = 'left')

    # ALTER
    dynamic_df = pd.merge(dynamic_df, drinks_per_week, left_on= 'sharealterid', right_index = True, how='left', suffixes = (None, '_alter')) 

    ########################## Add original DPW ##########################
    # import exams
    original_drinks_per_week = import_clinical_datasets.import_original_clinical_datasets_drinks_PW()
    # rename relevant exams to match offspring: 1-5
    renamed_dpw_original = original_drinks_per_week.rename(columns = {'total_12': 'total_1', 'total_17': 'total_2', 'total_19': 'total_3', 'total_21': 'total_4', 'total_23' : 'total_5'}) 
    # df with only relevant dpw:
    selected_dpw_original = renamed_dpw_original.loc[:,['total_1', 'total_2','total_3', 'total_4', 'total_5']]

    ############################### EGO ################################
    # Merge, with original getting the suffix total_x_y
    dynamic_df = dynamic_df.merge(selected_dpw_original, left_on = 'shareid', right_index = True, how = 'left', suffixes = (None, '_y'))
    # Fill merge total_x with total_x_y; fill na by the original value
    for c_name in ['total_1', 'total_2','total_3', 'total_4', 'total_5']:
        dynamic_df[c_name].fillna(dynamic_df[c_name+'_y'], inplace = True)
        dynamic_df.drop(c_name+'_y', axis=1, inplace=True)
    dynamic_df.reset_index(inplace = True, drop = True)
    ############################# ALTER ################################
    dynamic_df = dynamic_df.merge(selected_dpw_original, left_on = 'sharealterid', right_index = True, how = 'left', suffixes = (None, '_alter_y'))

    # Fill merge total_x with total_x_y; fill na by the original value
    for c_name in ['total_1_alter', 'total_2_alter','total_3_alter', 'total_4_alter', 'total_5_alter']:
        dynamic_df[c_name].fillna(dynamic_df[c_name+'_y'], inplace = True)
        dynamic_df.drop(c_name+'_y', axis=1, inplace=True)
    dynamic_df.reset_index(inplace = True, drop = True)
        
    ######################## CPD and AGE for offspring ######################
    # EGO
    meta = import_clinical_datasets.get_meta()
    meta = meta.filter(regex = '(CPD\d)|(AGE\d)')
    dynamic_df = pd.merge(dynamic_df, meta, left_on= 'shareid', right_index = True, how='left') 
    
    # ALTER 
    dynamic_df = pd.merge(dynamic_df, meta, left_on= 'sharealterid', right_index = True, how='left', suffixes = (None, '_alter')) 
    
    ######################## AGE for original #############################
    # EGO AGE
    ages_og = import_clinical_datasets.get_ages_original()
    dynamic_df = pd.merge(dynamic_df, ages_og, left_on= 'shareid', right_index = True, how='left', suffixes = (None, '_y'))
    for c_name in ['AGE1', 'AGE2','AGE3', 'AGE4', 'AGE5']:
        dynamic_df[c_name].fillna(dynamic_df[c_name+'_y'], inplace = True)
        dynamic_df.drop(c_name+'_y', axis=1, inplace=True)
    
    # ALTER AGE
    dynamic_df = pd.merge(dynamic_df, ages_og, left_on= 'sharealterid', right_index = True, how='left', suffixes = (None, '_alter_y'))
    for c_name in ['AGE1_alter', 'AGE2_alter','AGE3_alter', 'AGE4_alter', 'AGE5_alter']:
        dynamic_df[c_name].fillna(dynamic_df[c_name+'_y'], inplace = True)
        dynamic_df.drop(c_name+'_y', axis=1, inplace=True)

    return dynamic_df

#%%


def get_dates_org_off():
    """Obtains the dates at which each individual makes exams (offspring). Converts it into months.

    Returns:
        DataFrame: dateN with n from 1 to 7.
    """ 
    global dates   
    dates = import_clinical_datasets.get_dates_data()
    
    # Offspring
    dates_offspring = dates.loc[dates.idtype == 1]
    dates_in_months = (dates_offspring.filter(regex = '^(date[1-8]$)')*0.03285421).apply(np.ceil)
    dates_offspring = add_lapse(dates_in_months)
    dates_offspring = dates_offspring.reindex(columns = ['date1', 'date2', 'date3', 'date4', 'date5', 'date6', 'date7', 'date8'])
    
    # Original
    dates = import_clinical_datasets.get_dates_data()
    dates_org = dates.loc[dates.idtype == 0 ]
    dates_in_months = (dates_org*0.03285421).apply(np.ceil)
    dates_org = add_lapse(dates_in_months)
    dates_org = dates_org[['date12', 'date17','date19','date21','date23','date1']]
    dates_normalized = dates_org.add(dates_org.date1, axis = 'rows')
    dates_normalized.drop(columns = ['date1'], inplace = True)
    dates_org = dates_normalized.rename(columns = {'date12':'date1', 'date17':'date2','date19':'date3','date21':'date4','date23':'date5'})
    return dates_org, dates_offspring

#%%
def get_dynamic_personal_exam_match(dynamic_df):
    """Obtains the columns ['last_exam_before', 'month_diff_b4', 'earliest_exam_after', 'month_diff_after'] in dynamic df by comparing the moment(month) of making the exam for each ego with the SPELLEND and SPELLBEGIN to get the exam last made b4 the connection as well as the exam made 'immediately' after the connection ended. Also calculates the difference in months between the SPELLEND/SPELLBEGIN and those exams.

    Args:
        dynamic_df (_type_): _description_
    """    
    dates_org, dates_offspring = get_dates_org_off()
    
    def new_columns_dates(row): 
        """Goes along each row for each individual, looks at date df to see month in which exams were made and compares with SPELLBEGIN/SPELLEND. If connection was made b4 first or after last exam, it returns -1 or 99 for the exam number"""

        
        if row.idtype == 0:
            # dates_ego = next(iter([dates_org.loc[row.shareid].values]), [-1,99])
            try:
                dates_ego = dates_org.loc[row.shareid].values
            except KeyError:
                dates_ego = np.array([-1,99])
        
        if row.idtype == 1:
            # dates_ego = next(iter([dates_offspring.loc[row.shareid].values]), [-1,99])
            try:
                dates_ego = dates_offspring.loc[row.shareid].values
            except KeyError:
                dates_ego = np.array([-1,99])

            
        last_exam_before = np.where(dates_ego < row.SPELLBEGIN)[0].max(initial = -1)
        
        if last_exam_before == -1:
            month_diff_b4 = np.nan
        else:
            month_diff_b4 = row.SPELLBEGIN - dates_ego[last_exam_before] 
            
        first_exam_in = np.where(dates_ego > row.SPELLBEGIN)[0].min(initial = 99)
        if first_exam_in == 99:
            month_first_after  = np.nan
        else:
            month_first_after  = dates_ego[first_exam_in] - row.SPELLBEGIN

        
        earliest_exam_after = np.where(dates_ego > row.SPELLEND)[0].min(initial = 99)
        
        if earliest_exam_after == 99:
            month_diff_after = np.nan
        else:
            month_diff_after = dates_ego[earliest_exam_after] - row.SPELLEND
        
        return pd.Series([last_exam_before+1, month_diff_b4, first_exam_in+1, month_first_after, earliest_exam_after+1, month_diff_after])

    dynamic_df[['last_exam_before', 'month_diff_b4','first_exam_in', 'month_first_after',  'earliest_exam_after', 'month_diff_after']] = dynamic_df.apply(new_columns_dates, axis = 1)
    
    return dynamic_df

#%%

def split_dynamic_df_v2(dynamic_df):
    """Creates a list of dataframes containing only the connections for each wave.
    

    Args:
        dynamic_df (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df_list = []
    for wave_num in range(7):

        df = dynamic_df.copy()
        df = df.loc[df.last_exam_before < wave_num + 1]
        df = df.loc[df.earliest_exam_after > wave_num + 1]
        
        #! To avoid deprecated error:  use assign:
        #! df_list[i].assign(d_state_ego =  df_list[i][f"d_state_ego_{i+1}"])
        df['dpw'] = df[f"total_{wave_num+1}"]
        df['dpw_alter'] = df[f"total_{wave_num+1}_alter"]
        df['ego_trm'] = df[f"EGO_TREIMAN{wave_num+1}"]
        df['alter_trm'] = df[f"ALTER_TREIMAN{wave_num+1}"]
        df['age'] = df[f"AGE{wave_num+1}"]
        df['cpd'] = df[f"CPD{wave_num+1}"]
        df['d_state_ego'] = df[f"d_state_ego_{wave_num+1}"]
        df['d_state_alter'] = df[f"d_state_alter_{wave_num+1}"]
        df['alter_age'] = df[f"AGE{wave_num+1}_alter"]
        ####
        # df['cpd_alter'] = df[f"CPD{wave_num+1}_alter"]    
        ####
        df['wave'] = wave_num+1
        
        
        df = df.filter(regex = '^(?!(total_\d))')
        df = df.filter(regex= '^(?!(EGO_TREIMAN\d))')
        df = df.filter(regex= '^(?!(ALTER_TREIMAN\d))')
        df = df.filter(regex= '^(?!(CPD\d))')
        df = df.filter(regex= '^(?!(CPD\d_alter))')
        df = df.filter(regex= '^(?!(d_state_ego_\d))')
        df = df.filter(regex= '^(?!(d_state_alter_\d))')
        # df = df.filter(regex= '^(?!(AGE\d))')
        df = df.filter(regex= '^(?!(AGE\d))')
        df_list.append(df)
        
    return df_list
    

#%%


#%%
def add_sex(df):
    sex_df = import_clinical_datasets.get_sex_data().set_index('shareid')
    sex_df = sex_df.rename(columns = {'sex': 'sex_ego'})
    
    df = pd.merge(df, sex_df, left_on= 'shareid', right_index=True , how='left')
    
    sex_df = sex_df.rename(columns = {'sex_ego': 'sex_alter'})
    
    df = pd.merge(df, sex_df, left_on= 'sharealterid', right_index= True, how='left')
    
    return df
       
#%%

def get_drink_state(dynamic_df):
    """Categorizes the drinks per week into states. -1 for Na, 0 for abstain, 1 for normal and 2 for heavy. Since men and woman have different thresholds, it keeps that in mind

    Args:
        df_list (list): output of split_dynamic_df

    Returns:
        list: list with DF's with drink_state column attached
    """ 
    
    for i, column_name in enumerate(['total_1', 'total_2', 'total_3', 'total_4', 'total_5', 'total_6', 'total_7', 'total_8']):
        
        ##### EGO DRINKING STATE #######
        fem = dynamic_df.loc[dynamic_df.sex_ego == 2, ['shareid', column_name]]
        fem['d_state_ego_{}'.format(i+1)]= pd.cut(fem[column_name], bins =  [-10,-1,0,7,1000], labels = [-1,0,1,2])

        men = dynamic_df.loc[dynamic_df.sex_ego == 1, ['shareid', column_name]]
        men['d_state_ego_{}'.format(i+1)]= pd.cut(men[column_name], bins =  [-10,-1,0,14,1000], labels = [-1,0,1,2])
        
        # For missing sex, we still know that 0 = abstain and >14 is heavy for everyone, and 0-7 is medium
        nan = dynamic_df.loc[dynamic_df.sex_ego.isna(), ['shareid', column_name]]
        nan['d_state_ego_{}'.format(i+1)]= pd.cut(nan[column_name], bins =  [-10,-1,0,7,14,1000], labels = [-1,0,1,-1,2], ordered = False)

        ego_d_states = fem.append(men)
        ego_d_states = ego_d_states.append(nan)
        ego_d_states.drop(column_name, axis =1, inplace = True)
        
        # there could be multiple connections per person
        ego_d_states.drop_duplicates(inplace = True)
        
        dynamic_df = dynamic_df.merge(ego_d_states, left_on = 'shareid', right_on = 'shareid', how = 'outer')

    for i, column_name in enumerate(['total_1_alter', 'total_2_alter', 'total_3_alter', 'total_4_alter', 'total_5_alter', 'total_6_alter', 'total_7_alter', 'total_8_alter']):

        ###### ALTER DRINKING STATE ######
        fem = dynamic_df.loc[dynamic_df.sex_alter == 2, ['sharealterid', column_name]]
        fem['d_state_alter_{}'.format(i+1)]= pd.cut(fem[column_name], bins =  [-10,-1,0,7,1000], labels = [-1,0,1,2])

        men = dynamic_df.loc[dynamic_df.sex_alter == 1, ['sharealterid', column_name]]
        men['d_state_alter_{}'.format(i+1)]= pd.cut(men[column_name], bins =  [-10,-1,0,14,1000], labels = [-1,0,1,2])
        
        # For missing sex, we still know that 0 = abstain and >14 is heavy for everyone
        nan = dynamic_df.loc[dynamic_df.sex_alter.isna(), ['sharealterid', column_name]]
        nan['d_state_alter_{}'.format(i+1)]= pd.cut(nan[column_name], bins =  [-10,-1,0,7,14,1000], labels = [-1,0,1,-1,2], ordered = False)
        
        alter_d_states = fem.append(men)
        alter_d_states = alter_d_states.append(nan)
        alter_d_states.drop(column_name, axis =1, inplace = True)
        
        # there could be multiple connections per person
        alter_d_states.drop_duplicates(inplace = True)

        
        dynamic_df = dynamic_df.merge(alter_d_states, left_on = 'sharealterid', right_on = 'sharealterid', how = 'outer')
    
    return dynamic_df
       


#%%
def obtain_total_df(include = ['CHILD', 'SPOUSE', 'INGHBRNREL', 'SISTER', 'FRIENDNR', 'RELATIVENR',  'SAMEADNREL', 'MOTHER', 'BROTHER', 'FATHER']):
#, 'N100MNREL', 'N25MNREL', 'COWORKERNR']): 
    
    ############## Aggregate all data ###############
    
    # Obtain social network data
    sn = extract_social_net(include = include)
    
    # Add reciprocative friendship
    sn = reciprocate_friends(sn)
    
    # Add lapse
    sn = add_lapse(sn)

    # Obtain drinking, smoking, data, add it to social net
    dynamic_df = import_dpw_cpw_age(sn)

    # Add sex data
    dynamic_df = add_sex(dynamic_df)
    
    # Add the exam matching data
    dynamic_df = get_dynamic_personal_exam_match(dynamic_df)
    
    # Get drinking state columns
    dynamic_df = get_drink_state(dynamic_df)

    
    return dynamic_df
#%%


#%%
def get_df_list(dynamic_df):

    ############ Process of separate per wave ###########
 
    print('split into waves')
    df_list = split_dynamic_df_v2(dynamic_df)

    
    return  df_list

#%%
dynamic_df = obtain_total_df()   
#%%

df_list = get_df_list(dynamic_df)



#%%

# #%%


dynamic_df.to_csv('./data_files/main/df_nov.csv')
import pickle
with open('./data_files/main/df_list_nov.pkl', 'wb') as f:
    pickle.dump(df_list, f)

# # #%%
# df_list
# %%




















#def import_dpw_cpw_age(s_net):    
#     """imports the drinks per week from 'import__clinical_datasets' script, and merges it with the social network data. Further, it adds the data from the 'meta' file, containing CPD: cigarettes and Age per day. (only for offspring)
    
#     closest available moment for each offspring wave exam date. This corresponds to exam 12, 17, 21, 22, 23 -> in my own calculation:
#     # # def original_to_7_waves():
#     #     midpoints: [1972/3,1981,1989/90, 1993,1996/7, 1999/2000]
#     #     middle:    12,      17,   21 ,  22,  23,  24, 26
#     #     # closest to middle point, with alcohol data available
#     # 
    
#     But-> data says The time interval includes Original Cohort exams 12, 16, 19, 21, 23, 24, 26, 29 (https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/variable.cgi?study_id=phs000153.v9.p8&phv=73871)
#     #! 16 has no alcohol data so we take 17
#     # 
    
#     Returns:
#         DataFrame: social_net dataframe, with added columns total_x dpw for both original and offspring cohorts
#     """   
#     global dynamic_df
#     ########################## Add offspring DPW ###########################
#     drinks_per_week = import_clinical_datasets.import_clinical_datasets_drinks_PW()
#     dynamic_df = s_net.merge(drinks_per_week, left_on = 'shareid', right_index = True, how = 'left')
    
#     # ALTER
#     dynamic_df = pd.merge(dynamic_df, drinks_per_week, left_on= 'sharealterid', right_index = True, how='left', suffixes = (None, '_alter')) 

#     ########################## Add original DPW ##########################
#     # import exams
#     original_drinks_per_week = import_clinical_datasets.import_original_clinical_datasets_drinks_PW()
#     # rename relevant exams to match offspring: 1-5
#     renamed_dpw_original = original_drinks_per_week.rename(columns = {'total_12': 'total_1', 'total_17': 'total_2', 'total_19': 'total_3', 'total_21': 'total_4', 'total_23' : 'total_5'}) 
#     # df with only relevant dpw:
#     selected_dpw_original = renamed_dpw_original.loc[:,['total_1', 'total_2','total_3', 'total_4', 'total_5']]
#     # Merge, with original getting the suffix total_x_y
#     dynamic_df = dynamic_df.merge(selected_dpw_original, left_on = 'shareid', right_index = True, how = 'left', suffixes = (None, '_y'))
#     # Fill merge total_x with total_x_y; fill na by the original value
#     for c_name in ['total_1', 'total_2','total_3', 'total_4', 'total_5']:
#         dynamic_df[c_name].fillna(dynamic_df[c_name+'_y'], inplace = True)
#         dynamic_df.drop(c_name+'_y', axis=1, inplace=True)

#     dynamic_df.reset_index(inplace = True, drop = True)
    
#     ######################## CPD and AGE for offspring ######################
#     # EGO
#     meta = import_clinical_datasets.get_meta()
#     meta = meta.filter(regex = '(CPD\d)|(AGE\d)')
#     dynamic_df = pd.merge(dynamic_df, meta, left_on= 'shareid', right_index = True, how='left') 
    
#     # ALTER 
#     dynamic_df = pd.merge(dynamic_df, meta, left_on= 'sharealterid', right_index = True, how='left', suffixes = (None, '_alter')) 
    
#     ######################## AGE for original #############################
#     # EGO AGE
#     ages_og = import_clinical_datasets.get_ages_original()
#     dynamic_df = pd.merge(dynamic_df, ages_og, left_on= 'shareid', right_index = True, how='left', suffixes = (None, '_y'))
#     for c_name in ['AGE1', 'AGE2','AGE3', 'AGE4', 'AGE5']:
#         dynamic_df[c_name].fillna(dynamic_df[c_name+'_y'], inplace = True)
#         dynamic_df.drop(c_name+'_y', axis=1, inplace=True)
    
#     # ALTER AGE
#     dynamic_df = pd.merge(dynamic_df, ages_og, left_on= 'sharealterid', right_index = True, how='left', suffixes = (None, '_alter_y'))
#     for c_name in ['AGE1_alter', 'AGE2_alter','AGE3_alter', 'AGE4_alter', 'AGE5_alter']:
#         dynamic_df[c_name].fillna(dynamic_df[c_name+'_y'], inplace = True)
#         dynamic_df.drop(c_name+'_y', axis=1, inplace=True)

#     return dynamic_df


# !SHOULD BE DEPRECATED NOW
# def get_drink_state(df_list):
#     """Categorizes the drinks per week into states. -1 for Na, 0 for abstain, 1 for normal and 2 for heavy. Since men and woman have different thresholds, it keeps that in mind

#     Args:
#         df_list (list): output of split_dynamic_df

#     Returns:
#         list: list with DF's with drink_state column attached
#     """ 
       
#     dd2 = []
#     for i, wave in enumerate(df_list):
        
#         ##### EGO DRINKING STATE #######
#         fem = wave.loc[wave.sex_ego == 2, ['shareid', 'dpw']]
#         fem['d_state_ego']= pd.cut(fem['dpw'], bins =  [-10,-1,0,7,1000], labels = [-1,0,1,2])

#         men = wave.loc[wave.sex_ego == 1, ['shareid', 'dpw']]
#         men['d_state_ego']= pd.cut(men['dpw'], bins =  [-10,-1,0,14,1000], labels = [-1,0,1,2])

#         all_drink_states = fem.append(men)
#         all_drink_states.drop('dpw', axis =1, inplace = True)
        
#         # there could be multiple connections per person
#         all_drink_states.drop_duplicates(inplace = True)

#         wave2 = pd.merge(wave, all_drink_states, left_on = 'shareid', right_on = 'shareid', how = 'left')
        
#         ###### ALTER DRINKING STATE ######
#         fem = wave.loc[wave.sex_alter == 2, ['sharealterid', 'dpw_alter']]
#         fem['d_state_alter']= pd.cut(fem['dpw_alter'], bins =  [-10,-1,0,7,1000], labels = [-1,0,1,2])

#         men = wave.loc[wave.sex_alter == 1, ['sharealterid', 'dpw_alter']]
#         men['d_state_alter']= pd.cut(men['dpw_alter'], bins =  [-10,-1,0,14,1000], labels = [-1,0,1,2])

#         all_drink_states = fem.append(men)
#         all_drink_states.drop('dpw_alter', axis =1, inplace = True)
        
#         # there could be multiple connections per person
#         all_drink_states.drop_duplicates(inplace = True)

#         wave3 = pd.merge(wave2, all_drink_states, left_on = 'sharealterid', right_on = 'sharealterid', how = 'left')
        
#         dd2.append(wave3)
          
#     return dd2    

#%%
# *SHould be deprecated now that i do it in dynamic df get cpw age 
# def get_alter_dpw(df_list):
#     def get_dpw_row(row):
#         value_list = wave.loc[wave.shareid == row.sharealterid].dpw.values
#         value =  next(iter(value_list), np.nan)
#         return pd.Series([value])
    
#     for wave in df_list:
#         wave[['dpw_alter']] = wave.apply(get_dpw_row, axis = 1)
        
#     return df_list
