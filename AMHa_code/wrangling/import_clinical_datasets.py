#%%
import pandas as pd
from pathlib import Path

# data_folder = Path("~/Downloads/sratoolkit.2.11.1-ubuntu64/bin/dbGaP-27276/")
data_folder = '../'/Path.home()/'Documents/HPC_treasure/hpc/sratoolkit.2.11.1-ubuntu64/bin/dbGaP-27276/'


def import_clinical_datasets_drinks_PW():
    # exam 1:    A111: beer intake, A112 wine, A113, cocktails, A115 = AMOUNT OF ALCOHOL (more or less than last 2 days)
    #? Cutoff date: 1971 (t/m) 1975 = 5 jaar = 60 maanden + 1 want start at 1 ipv 0 -> 61
    exam_1_c1 = data_folder/"./offspring_exams/phs000007.v32.pht000030.v9.p13.c1.ex1_1s.HMB-IRB-MDS.txt"
    exam_1_c2 = data_folder/"./offspring_exams/phs000007.v32.pht000030.v9.p13.c2.ex1_1s.HMB-IRB-NPU-MDS.txt"
    df_1_1 = pd.read_csv(exam_1_c1, skiprows=10, delimiter='\t')
    df_1_2 = pd.read_csv(exam_1_c2, skiprows=10, delimiter='\t')
    df_1 = df_1_1.append(df_1_2)[['shareid','A111','A112','A113']].set_index('shareid')
    df_1["total_1"] = df_1['A111']+df_1['A112']+df_1['A113']
    
    
    # B117 #BEER PER WEEK, B118 #WINE PER WEEK, #B119 COCKTAILS PER WEEK
    # B120 DAYS OF BEER PER WEEK, B121 DAYS OF WINE PER WEEK, B122 COCKTAILS
    # B123 LIMIT PER TIME BEER, B124 WINE, B125 COCKTAILS
    #? Cutoff date: 1979-82, 82-71 -> 144 -> 145 maanden, session time: 144-60 = 84 
    exam_2_c1 = data_folder/"./offspring_exams/phs000007.v32.pht000031.v9.p13.c1.ex1_2s.HMB-IRB-MDS.txt"
    exam_2_c2 = data_folder/"./offspring_exams/phs000007.v32.pht000031.v9.p13.c2.ex1_2s.HMB-IRB-NPU-MDS.txt"
    df_2_1 = pd.read_csv(exam_2_c1, skiprows=10, delimiter='\t')
    df_2_2 = pd.read_csv(exam_2_c2, skiprows=10, delimiter='\t')
    df_2 = df_2_1.append(df_2_2)[['shareid','B117','B118','B119']].set_index('shareid')
    df_2["total_2"] = df_2['B117']+df_2['B118']+df_2['B119']
    

    # C83: BEER: BOTTLES, CANS, GLASSES PER WEEK, C86 WINE, C89 DRINKS WITH LIQUOR
    # C84: BEER: NO. OF DAYS PER WEEK DRINK BEER, C87 WINE, C90 DRINKS WITH LIQUOR
    # C85: BEER: LIMIT AT ONE PERIOD OF TIME,     C88 WINE, C91 DRINKS WITH LIQUOR
    #? Cutoff date: 84-87 -> 204 -> 205 maanden, session time: 204-144 = 60
    exam_3_c1 = data_folder/"./offspring_exams/phs000007.v32.pht000032.v8.p13.c1.ex1_3s.HMB-IRB-MDS.txt"
    exam_3_c2 = data_folder/"./offspring_exams/phs000007.v32.pht000032.v8.p13.c2.ex1_3s.HMB-IRB-NPU-MDS.txt"
    df_3_1 = pd.read_csv(exam_3_c1, skiprows=10, delimiter='\t')
    df_3_2 = pd.read_csv(exam_3_c2, skiprows=10, delimiter='\t')
    df_3 = df_3_1.append(df_3_2)[['shareid','C83','C86','C89']].set_index('shareid')
    df_3["total_3"] = df_3['C83']+df_3['C86']+df_3['C89']

    # D082: BEER: BOTTLES, CANS, GLASSES PER WEEK, D085 WINE, D088 COCKTAILS
    # D083: BEER: NO. OF DAYS PER WEEK DRINK BEER, D086 WINE, D089 COCKTAILS
    # D084: BEER: LIMIT AT ONE PERIOD OF TIME,     D087 WINE, D090 COCKTAILS
    #? Cutoff date: 87-90 -> 240 -> 241 maanden, session time: 240 - 204 = 60
    exam_4_c1 = data_folder/"./offspring_exams/phs000007.v32.pht000033.v10.p13.c1.ex1_4s.HMB-IRB-MDS.txt"
    exam_4_c2 = data_folder/"./offspring_exams/phs000007.v32.pht000033.v10.p13.c2.ex1_4s.HMB-IRB-NPU-MDS.txt"
    df_4_1 = pd.read_csv(exam_4_c1, skiprows=10, delimiter='\t')
    df_4_2 = pd.read_csv(exam_4_c2, skiprows=10, delimiter='\t')
    df_4 = df_4_1.append(df_4_2)[['shareid','D082','D085','D088']].set_index('shareid')
    df_4["total_4"] = df_4['D082']+df_4['D085']+df_4['D088']



    # SAME METHOD, STARTING AT: E310 FOR BEER # PER WEEK, UNTIL E318 
    #? Cutoff date: 91-95 -> 300 -> 301 maanden, 300-240 = 60
    exam_5_c1 = data_folder/"./offspring_exams/phs000007.v32.pht000034.v9.p13.c1.ex1_5s.HMB-IRB-MDS.txt"
    exam_5_c2 = data_folder/"./offspring_exams/phs000007.v32.pht000034.v9.p13.c2.ex1_5s.HMB-IRB-NPU-MDS.txt"
    df_5_1 = pd.read_csv(exam_5_c1, skiprows=10, delimiter='\t')
    df_5_2 = pd.read_csv(exam_5_c2, skiprows=10, delimiter='\t')
    df_5 = df_5_1.append(df_5_2)[['shareid','E310','E313','E316']].set_index('shareid')
    df_5["total_5"] = df_5['E310']+df_5['E313']+df_5['E316']

    # F276 TILL F278 voor bier, dan is er verschil witte en rode wine:
    # F279, 280 en 281 zijn witte wijn, dan F282, F283, F284 rode wijn
    # F285, 286,287 voor cocktails
    #? Cutoff date: 96-98 -> 336 -> 337 maanden, 336-300 = 36
    exam_6_c1 = data_folder/"./offspring_exams/phs000007.v32.pht000035.v10.p13.c1.ex1_6s.HMB-IRB-MDS.txt"
    exam_6_c2 = data_folder/"./offspring_exams/phs000007.v32.pht000035.v10.p13.c2.ex1_6s.HMB-IRB-NPU-MDS.txt"
    df_6_1 = pd.read_csv(exam_6_c1, skiprows=10, delimiter='\t')
    df_6_2 = pd.read_csv(exam_6_c2, skiprows=10, delimiter='\t')
    df_6 = df_6_1.append(df_6_2)[['shareid','F276','F279','F282','F285']].set_index('shareid')
    df_6["total_6"] = df_6['F276']+df_6['F279']+ df_6['F282'] + df_6['F285']

    
    # G104, G105, G106 BIER, 7,8,9 witte wijn, 10,11,12 rode 13,14,15 cocktails
    #? Cutoff date: 98-01 -> 372 -> 373 maanden, 372-336 = 36
    exam_7_c1 = data_folder/"./offspring_exams/phs000007.v32.pht000036.v10.p13.c1.ex1_7s.HMB-IRB-MDS.txt"
    exam_7_c2 = data_folder/"./offspring_exams/phs000007.v32.pht000036.v10.p13.c2.ex1_7s.HMB-IRB-NPU-MDS.txt"
    df_7_1 = pd.read_csv(exam_7_c1, skiprows=10, delimiter='\t')
    df_7_2 = pd.read_csv(exam_7_c2, skiprows=10, delimiter='\t')
    df_7 = df_7_1.append(df_7_2)[['shareid','G104','G107','G110', 'G113']].set_index('shareid')
    df_7["total_7"] = df_7['G104']+df_7['G107']+df_7['G110']+df_7['G113']

    # !EXAM 8 IS COMPLICATED AND DIFFERENT
    # To calculate total alcohol consumption H072, H073, H075, H076, H078 and H079 (sommige per maand sommige per week (--.--)
    #  H081 is how many days per week did you drink avg, en H082 is how many drinks do you do on day where u drink
    #? Cutoff date: 2005-2008 -> 456 -> 457 maanden, 456-372 = 84
    exam_8_c1 = data_folder/"./offspring_exams/phs000007.v32.pht000747.v7.p13.c1.ex1_8s.HMB-IRB-MDS.txt"
    exam_8_c2 = data_folder/"./offspring_exams/phs000007.v32.pht000747.v7.p13.c2.ex1_8s.HMB-IRB-NPU-MDS.txt"
    df_8_1 = pd.read_csv(exam_8_c1, skiprows=10, delimiter='\t')
    df_8_2 = pd.read_csv(exam_8_c2, skiprows=10, delimiter='\t')
    #%%
    df_8 = df_8_1.append(df_8_2)[['shareid','H081', 'H082']].set_index('shareid')
    df_8["total_8"] = df_8['H081'] * df_8['H082']

    df_final = df_1.merge(df_2["total_2"], left_index= True, right_index= True, how= 'outer')[["total_1","total_2"]]
    #%%
    # ! STILL NEED TO DEAL WITH PPL UNDER DRINKING AGE!
    df_final = df_final.merge(df_3["total_3"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_4["total_4"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_5["total_5"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_6["total_6"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_7["total_7"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_8["total_8"], left_index= True, right_index= True, how= 'outer') 
    # + df_3["total_3"] + df_4["total_4"] + df_5["total_5"] + df_6["total_6"] + df_7["total_7"] + df_8["total_8"] 

    return df_final
# %%
def import_original_clinical_datasets_drinks_PW():
    # exam 1:    FE127: beer intake, FE128 wine, FE129, cocktails, P/W (geen info over binge/#dagen)
    #? Cutoff date: 1971 (t/m) 1974 
    exam_12_c1 = data_folder/"./original_exams/phs000007.v32.pht000014.v3.p13.c1.ex0_12s.HMB-IRB-MDS.txt"
    exam_12_c2 = data_folder/"./original_exams/phs000007.v32.pht000014.v3.p13.c2.ex0_12s.HMB-IRB-NPU-MDS.txt"
    df_12_1 = pd.read_csv(exam_12_c1, skiprows=10, delimiter='\t')
    df_12_2 = pd.read_csv(exam_12_c2, skiprows=10, delimiter='\t')
    df_12 = df_12_1.append(df_12_2)[['shareid','FE127','FE128','FE129']].set_index('shareid')
    df_12["total_12"] = df_12['FE127']+df_12['FE128']+df_12['FE129']

    exam_13_c1 = data_folder/"./original_exams/phs000007.v32.pht000015.v3.p13.c1.ex0_13s.HMB-IRB-MDS.txt"
    exam_13_c2 = data_folder/"./original_exams/phs000007.v32.pht000015.v3.p13.c2.ex0_13s.HMB-IRB-NPU-MDS.txt"
    df_13_1 = pd.read_csv(exam_13_c1, skiprows=10, delimiter='\t')
    df_13_2 = pd.read_csv(exam_13_c2, skiprows=10, delimiter='\t')
    df_13 = df_13_1.append(df_13_2)[['shareid','FF125','FF126','FF127']].set_index('shareid')
    df_13["total_13"] = df_13['FF125']+df_13['FF126']+df_13['FF127']

    exam_14_c1 = data_folder/"./original_exams/phs000007.v32.pht000016.v3.p13.c1.ex0_14s.HMB-IRB-MDS.txt"
    exam_14_c2 = data_folder/"./original_exams/phs000007.v32.pht000016.v3.p13.c2.ex0_14s.HMB-IRB-NPU-MDS.txt" 
    df_14_1 = pd.read_csv(exam_14_c1, skiprows=10, delimiter='\t')
    df_14_2 = pd.read_csv(exam_14_c2, skiprows=10, delimiter='\t')
    df_14 = df_14_1.append(df_14_2)[['shareid','FG118','FG119','FG120']].set_index('shareid')
    df_14["total_14"] = df_14['FG118']+df_14['FG119']+df_14['FG120']

    exam_15_c1 = data_folder/"./original_exams/phs000007.v32.pht000017.v3.p13.c1.ex0_15s.HMB-IRB-MDS.txt"
    exam_15_c2 = data_folder/"./original_exams/phs000007.v32.pht000017.v3.p13.c2.ex0_15s.HMB-IRB-NPU-MDS.txt"
    df_15_1 = pd.read_csv(exam_15_c1, skiprows=10, delimiter='\t')
    df_15_2 = pd.read_csv(exam_15_c2, skiprows=10, delimiter='\t')
    df_15 = df_15_1.append(df_15_2)[['shareid','FH115','FH116','FH117']].set_index('shareid')
    df_15["total_15"] = df_15['FH115']+df_15['FH116']+df_15['FH117']

    #! Exam 16 no alcohol / nicotine data

    exam_17_c1 = data_folder/"./original_exams/phs000007.v32.pht000019.v3.p13.c1.ex0_17s.HMB-IRB-MDS.txt"
    exam_17_c2 = data_folder/"./original_exams/phs000007.v32.pht000019.v3.p13.c2.ex0_17s.HMB-IRB-NPU-MDS.txt"
    df_17_1 = pd.read_csv(exam_17_c1, skiprows=10, delimiter='\t')
    df_17_2 = pd.read_csv(exam_17_c2, skiprows=10, delimiter='\t')
    df_17 = df_17_1.append(df_17_2)[['shareid','FJ59','FJ60','FJ61']].set_index('shareid')
    df_17["total_17"] = df_17['FJ59']+df_17['FJ60']+df_17['FJ61']

    exam_18_c1 = data_folder/"./original_exams/phs000007.v32.pht000020.v3.p13.c1.ex0_18s.HMB-IRB-MDS.txt"
    exam_18_c2 = data_folder/"./original_exams/phs000007.v32.pht000020.v3.p13.c2.ex0_18s.HMB-IRB-NPU-MDS.txt"
    df_18_1 = pd.read_csv(exam_18_c1, skiprows=10, delimiter='\t')
    df_18_2 = pd.read_csv(exam_18_c2, skiprows=10, delimiter='\t')
    df_18 = df_18_1.append(df_18_2)[['shareid','FK141','FK142','FK143']].set_index('shareid')
    df_18["total_18"] = df_18['FK141']+df_18['FK142']+df_18['FK143']

    
    # Nu gaat hij ook beer -> per week, # per week van drinken, hoeveel per keer 
    exam_19_c1 = data_folder/"./original_exams/phs000007.v32.pht000021.v3.p13.c1.ex0_19s.HMB-IRB-MDS.txt"
    exam_19_c2 = data_folder/"./original_exams/phs000007.v32.pht000021.v3.p13.c2.ex0_19s.HMB-IRB-NPU-MDS.txt"
    df_19_1 = pd.read_csv(exam_19_c1, skiprows=10, delimiter='\t')
    df_19_2 = pd.read_csv(exam_19_c2, skiprows=10, delimiter='\t')
    df_19 = df_19_1.append(df_19_2)[['shareid','FL202','FL205','FL208']].set_index('shareid')
    df_19["total_19"] = df_19['FL202']+df_19['FL205']+df_19['FL208']
    
    exam_20_c1 = data_folder/"./original_exams/phs000007.v32.pht000022.v4.p13.c1.ex0_20s.HMB-IRB-MDS.txt"
    exam_20_c2 = data_folder/"./original_exams/phs000007.v32.pht000022.v4.p13.c2.ex0_20s.HMB-IRB-NPU-MDS.txt"
    df_20_1 = pd.read_csv(exam_20_c1, skiprows=10, delimiter='\t')
    df_20_2 = pd.read_csv(exam_20_c2, skiprows=10, delimiter='\t')
    df_20 = df_20_1.append(df_20_2)[['shareid','FM221','FM224','FM227']].set_index('shareid')
    df_20["total_20"] = df_20['FM221']+df_20['FM224']+df_20['FM227']
    
    exam_21_c1 = data_folder/"./original_exams/phs000007.v32.pht000023.v4.p13.c1.ex0_21s.HMB-IRB-MDS.txt"
    exam_21_c2 = data_folder/"./original_exams/phs000007.v32.pht000023.v4.p13.c2.ex0_21s.HMB-IRB-NPU-MDS.txt"
    df_21_1 = pd.read_csv(exam_21_c1, skiprows=10, delimiter='\t')
    df_21_2 = pd.read_csv(exam_21_c2, skiprows=10, delimiter='\t')
    df_21 = df_21_1.append(df_21_2)[['shareid','FN175','FN178','FN181']].set_index('shareid')
    df_21["total_21"] = df_21['FN175']+df_21['FN178']+df_21['FN181']
    
    
    exam_22_c1 = data_folder/"./original_exams/phs000007.v32.pht000024.v5.p13.c1.ex0_22s.HMB-IRB-MDS.txt"
    exam_22_c2 = data_folder/"./original_exams/phs000007.v32.pht000024.v5.p13.c2.ex0_22s.HMB-IRB-NPU-MDS.txt"
    df_22_1 = pd.read_csv(exam_22_c1, skiprows=10, delimiter='\t')
    df_22_2 = pd.read_csv(exam_22_c2, skiprows=10, delimiter='\t')
    df_22 = df_22_1.append(df_22_2)[['shareid','FO171','FO174','FO177']].set_index('shareid')
    df_22["total_22"] = df_22['FO171']+df_22['FO174']+df_22['FO177']
    
    
    exam_23_c1 = data_folder/"./original_exams/phs000007.v32.pht000025.v4.p13.c1.ex0_23s.HMB-IRB-MDS.txt"
    exam_23_c2 = data_folder/"./original_exams/phs000007.v32.pht000025.v4.p13.c2.ex0_23s.HMB-IRB-NPU-MDS.txt"
    df_23_1 = pd.read_csv(exam_23_c1, skiprows=10, delimiter='\t')
    df_23_2 = pd.read_csv(exam_23_c2, skiprows=10, delimiter='\t')
    df_23 = df_23_1.append(df_23_2)[['shareid','FP111','FP114','FP117']].set_index('shareid')
    df_23["total_23"] = df_23['FP111']+df_23['FP114']+df_23['FP117']
    
    
    
    #! exam 24 en 25 hebben geen alcohol data  (! Wel nicotine / smoking/ cigarette)
    # exam 26 is andere soort data (over vorig jaar, heb je ooit bier gedronken)
    

    df_final = df_12.merge(df_13["total_13"], left_index= True, right_index= True, how= 'outer')[["total_12","total_13"]]
    #%%
    # ! STILL NEED TO DEAL WITH PPL UNDER DRINKING AGE!
    df_final = df_final.merge(df_14["total_14"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_15["total_15"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_17["total_17"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_18["total_18"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_19["total_19"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_20["total_20"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_21["total_21"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_22["total_22"], left_index= True, right_index= True, how= 'outer') 
    df_final = df_final.merge(df_23["total_23"], left_index= True, right_index= True, how= 'outer') 
    # + df_3["total_3"] + df_4["total_4"] + df_5["total_5"] + df_6["total_6"] + df_7["total_7"] + df_8["total_8"] 

    return df_final
# %%
original_drinking = import_original_clinical_datasets_drinks_PW()
# %%

def get_dates_data():
    dates_csv_1 = data_folder/'./meta_info/phs000007.v32.pht003099.v7.p13.c1.vr_dates_2019_a_1175s.HMB-IRB-MDS.txt'
    dates_csv_2 = data_folder/'./meta_info/phs000007.v32.pht003099.v7.p13.c2.vr_dates_2019_a_1175s.HMB-IRB-NPU-MDS.txt'
    dates_df_1 = pd.read_csv(dates_csv_1, skiprows=10, delimiter='\t')
    dates_df_2 = pd.read_csv(dates_csv_2, skiprows=10, delimiter='\t')
    dates_df = dates_df_1.append(dates_df_2).set_index('shareid')
    dates_df = dates_df.loc[dates_df.idtype < 3]
    dates_df = dates_df.filter(regex = '^(?!(att\d))')
    dates_df.drop(columns='dbGaP_Subject_ID',inplace=True)
    return dates_df

#%%
def get_meta():
    """Gets offspring meta information, from an aggregate file

    Returns:
        _type_: _description_
    """    
    offspring_c1 = data_folder/"meta_info/phs000007.v32.pht006027.v3.p13.c1.vr_wkthru_ex09_1_1001s.HMB-IRB-MDS.txt"
    offspring_c2 = data_folder/"meta_info/phs000007.v32.pht006027.v3.p13.c2.vr_wkthru_ex09_1_1001s.HMB-IRB-NPU-MDS.txt"

    offspring_c1 = pd.read_csv(offspring_c1, skiprows=10, delimiter='\t')
    offspring_c2 = pd.read_csv(offspring_c2, skiprows=10, delimiter='\t')
    meta = offspring_c1.append(offspring_c2).set_index('shareid')
    return meta

foo = get_meta()

# %%

def get_sex_data():
    loc = data_folder/'./social_net/phs000153.v9.pht000835.v8.p8.shareped_Pedigree.MULTI.txt'
    sex_df = pd.read_csv(loc, skiprows=10, delimiter='\t')
    sex_df = sex_df[['shareid','sex']]
    return sex_df

# %%
def get_ages_original():
    dates_csv_1 = data_folder/'./meta_info/phs000007.v32.pht003099.v7.p13.c1.vr_dates_2019_a_1175s.HMB-IRB-MDS.txt'
    dates_csv_2 = data_folder/'./meta_info/phs000007.v32.pht003099.v7.p13.c2.vr_dates_2019_a_1175s.HMB-IRB-NPU-MDS.txt'
    dates_df_1 = pd.read_csv(dates_csv_1, skiprows=10, delimiter='\t')
    dates_df_2 = pd.read_csv(dates_csv_2, skiprows=10, delimiter='\t')
    dates_df = dates_df_1.append(dates_df_2).set_index('shareid')
    age_df = dates_df[['age12','age17','age19','age21','age23']]
    age_df = age_df.rename(columns = {'age12':'AGE1','age17':'AGE2','age19':'AGE3','age21':'AGE4','age23':'AGE5'})
    return age_df

# age_df = get_ages_original()
# %%
