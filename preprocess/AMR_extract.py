# %%
import pandas as pd 
import os 
# %%
ed_path = '/mnt/bigdata/compmed/physionet/mimic-iv-ed-2.2/mimic-iv-ed-2.2/'
hosp_path = '/mnt/bigdata/compmed/physionet/mimic-iv-2.2/mimic-iv-2.2/'
# %%
# cohort
edstays = pd.read_csv(os.path.join(ed_path, 'ed/edstays.csv.gz'), compression='gzip')
# edstays = edstays[edstays['disposition'].isin(valid_dispositions)]
# external info
# GET AGE
edstays = edstays.merge(
    pd.read_csv(os.path.join(hosp_path, 'hosp/patients.csv.gz'), compression='gzip')\
        [['subject_id', 'anchor_age', 'dod']], 
        on='subject_id', how='left')
# %%
# microbiology
microbiology = pd.read_csv(os.path.join(hosp_path, 'hosp/microbiologyevents.csv.gz'), compression='gzip')
# only patients who were admitted
microbiology = microbiology.dropna(subset=['org_name', 'ab_name'])\
    .merge(edstays[['subject_id', 'hadm_id']].drop_duplicates(), on=['subject_id', 'hadm_id'], how='inner')
print(microbiology.shape)

# only consider patients who had cultures done DURING admission
microbiology = microbiology[[
    'subject_id', 'hadm_id'
    , 'chartdate', 'charttime'
    , 'test_name', 'org_name', 'ab_name'
    , 'interpretation']].dropna(subset='hadm_id')

# top 20 - later look into e. coli and klebsiella and pseudomonas
# ESCHERICHIA COLI                                   50941
# STAPH AUREUS COAG +                                40337
# KLEBSIELLA PNEUMONIAE                              18343
# PSEUDOMONAS AERUGINOSA                             15329
# ENTEROCOCCUS SP.                                   10329
# STAPHYLOCOCCUS, COAGULASE NEGATIVE                  7972
# PROTEUS MIRABILIS                                   7846
# YEAST                                               7656
# ENTEROBACTER CLOACAE COMPLEX                        5256
# STAPHYLOCOCCUS EPIDERMIDIS                          3463
# KLEBSIELLA OXYTOCA                                  3311
# MIXED BACTERIAL FLORA                               3032
# SERRATIA MARCESCENS                                 2962
# ENTEROCOCCUS FAECIUM                                2482
# ENTEROBACTER AEROGENES                              2001
# STREPTOCOCCUS ANGINOSUS (MILLERI) GROUP             1882
# POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS     1868
# CLOSTRIDIUM DIFFICILE                               1800
# CANCELLED                                           1791
# CITROBACTER FREUNDII COMPLEX                        1697
# Name: org_name, dtype: int64

# %%
# limit to staph to be consistent with trevor
microbiology = microbiology[microbiology.org_name.str.contains('STAPH')]
mapping = {'S': 1, 'I': 0, 'R': 0, 'P': 0}
microbiology['Susceptibility'] = microbiology.interpretation.map(mapping)\
    .fillna(0) # assume resistant/ineffective if tested and other value present
# %%
personalized_antibiogram = microbiology\
    .groupby([
        'subject_id', 'hadm_id'
        , 'chartdate', 'charttime'
        , 'test_name', 'org_name'
        , 'ab_name'], as_index=False)\
    ['Susceptibility'].min()\
    .pivot(
        columns='ab_name'
        , values='Susceptibility'
        , index=['subject_id', 'hadm_id', 'chartdate', 'charttime', 'test_name', 'org_name']
    )\
    [['CLINDAMYCIN', 'DAPTOMYCIN', 'ERYTHROMYCIN'
      , 'GENTAMICIN', 'LEVOFLOXACIN', 'NITROFURANTOIN'
      , 'OXACILLIN', 'RIFAMPIN', 'TETRACYCLINE'
      , 'TRIMETHOPRIM/SULFA', 'VANCOMYCIN']]
    # AMPICILLIN            6121
    # CIPROFLOXACIN         6121
    # CLINDAMYCIN            488
    # DAPTOMYCIN            5912
    # ERYTHROMYCIN           469
    # GENTAMICIN              14
    # IMIPENEM              6124
    # LEVOFLOXACIN           152
    # LINEZOLID             6011
    # MEROPENEM             6124
    # NITROFURANTOIN        5725
    # OXACILLIN               18
    # PENICILLIN G          6118
    # RIFAMPIN              3614
    # TETRACYCLINE           969
    # TRIMETHOPRIM/SULFA    1322
    # VANCOMYCIN            2927
# NOTE: NaN means not tested. need to decide if filling with 0 (assuming resistant) is appropriate.
# %%
### ### 
# PRESCRIPTIONS 

prescription = pd.read_csv(os.path.join(hosp_path, 'hosp/prescriptions.csv.gz'), compression='gzip')\
    .dropna(subset=['drug', 'hadm_id'])\
    .merge(edstays[['hadm_id', 'subject_id']].drop_duplicates(), on=['subject_id', 'hadm_id'], how='inner')
# %%
# only antibiotics
ab_list = ['CLINDAMYCIN', 'DAPTOMYCIN', 'ERYTHROMYCIN'
      , 'GENTAMICIN', 'LEVOFLOXACIN', 'NITROFURANTOIN'
      , 'OXACILLIN', 'RIFAMPIN', 'TETRACYCLINE'
      , 'TRIMETHOPRIM/SULFA', 'VANCOMYCIN']

abx_df_list = []
for ab in ab_list:
    ab_df = prescription[prescription.drug.str.upper().str.contains(ab)]
    ab_df['ab_name'] = ab 
    abx_df_list.append(ab_df)

abx_prescriptions = pd.concat(abx_df_list)
print(abx_prescriptions.shape)
# NOTE: ALL patients who had antibiotics
# need to intersect with patients who also had cultures
# %%
