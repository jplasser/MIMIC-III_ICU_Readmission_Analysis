import csv
import numpy as np
import os
import pandas as pd
import sys

from mimic3benchmark.util import *

def read_patients_table(mimic4_path):
    # subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod
    # https://mimic-iv.mit.edu/docs/datasets/core/patients/
    # "ROW_ID","SUBJECT_ID","GENDER","DOB","DOD","DOD_HOSP","DOD_SSN","EXPIRE_FLAG"
    # old: https://mimic.physionet.org/mimictables/patients/
    pats = dataframe_from_csv(os.path.join(mimic4_path, 'patients.csv'), index_col=None)
    # this table is missing the DOB (date of birth), but has an age value in the form of anchor_age
    #pats = pats[['subject_id', 'gender', 'DOB', 'dod']]
    pats = pats[['subject_id', 'gender', 'anchor_age', 'dod']]
    #pats.DOB = pd.to_datetime(pats.DOB) # DOB is not available directly, we have to compute a DOB time, but we maybe can directly use the age
    pats.anchor_age = pd.to_numeric(pats.anchor_age) #.as_type(int)
    pats.dod = pd.to_datetime(pats.dod)
    return pats

def read_admissions_table(mimic4_path):
    # subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,insurance,language,marital_status,ethnicity,edregtime,edouttime,hospital_expire_flag
    # https://mimic-iv.mit.edu/docs/datasets/core/admissions/
    # "ROW_ID","SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME","DEATHTIME","ADMISSION_TYPE","ADMISSION_LOCATION","DISCHARGE_LOCATION","INSURANCE","LANGUAGE","RELIGION","MARITAL_STATUS","ETHNICITY","EDREGTIME","EDOUTTIME","DIAGNOSIS","HOSPITAL_EXPIRE_FLAG","HAS_CHARTEVENTS_DATA"
    # old: https://mimic.physionet.org/mimictables/admissions/
    admits = dataframe_from_csv(os.path.join(mimic4_path, 'admissions.csv'), index_col=None)
    #admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'insurance', 'religion', 'marital_status', 'ethnicity', 'diagnosis']]
    # diagnosis is not available here
    admits = admits[
        ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'insurance', 'marital_status',
         'ethnicity']]  # , 'diagnosis'
    # religion is missing in MIMIC-IV
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    return admits

def read_icustays_table(mimic4_path):
    # subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime,los
    # https://mimic-iv.mit.edu/docs/datasets/icu/icustays/
    # "ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","DBSOURCE","FIRST_CAREUNIT","LAST_CAREUNIT","FIRST_WARDID","LAST_WARDID","INTIME","OUTTIME","LOS"
    # old: https://mimic.physionet.org/mimictables/transfers/
    stays = dataframe_from_csv(os.path.join(mimic4_path, 'icustays.csv'), index_col=None)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays

def read_transfers_table(mimic4_path):
    # subject_id,hadm_id,transfer_id,eventtype,careunit,intime,outtime
    # https://mimic-iv.mit.edu/docs/datasets/core/transfers/
    # "ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","DBSOURCE","EVENTTYPE","PREV_CAREUNIT","CURR_CAREUNIT","PREV_WARDID","CURR_WARDID","INTIME","OUTTIME","LOS"
    # old: https://mimic.physionet.org/mimictables/transfers/
    transfers = dataframe_from_csv(os.path.join(mimic4_path + '/core', 'transfers.csv'), index_col=None)
    transfers.intime = pd.to_datetime(transfers.intime)
    transfers.outtime = pd.to_datetime(transfers.outtime)

    # we have to merge with some columns of the icustays table
    #stays = dataframe_from_csv(os.path.join(mimic4_path + '/icu', 'icustays.csv'), index_col=None)
    #stays.intime = pd.to_datetime(stays.intime)
    #stays.outtime = pd.to_datetime(stays.outtime)
    #stays = stays[["subject_id", "hadm_id", "stay_id"]]
    #transfers = transfers.merge(stays, how='left', left_on=['transfer_id'], right_on=['stay_id'])
    transfers['stay_id'] = transfers['transfer_id']

    # there's no ICUSTAY_ID column, but different ones that comprise (hopefully) the same data, took stay_id for now
    # TODO
    transfersnotnull = transfers.loc[transfers.stay_id.notnull()]
    #print(transfersnotnull)
    transfersisnull = transfers.loc[transfers.stay_id.isnull()]

    transfersnotnull=transfersnotnull.drop_duplicates('stay_id',keep='last')
    #print(transfersnotnull)
    transfers=pd.concat([transfersnotnull, transfersisnull])
    return transfers

def read_icd_diagnoses_table(mimic4_path):
    # icd_code,icd_version,long_title
    # https://mimic-iv.mit.edu/docs/datasets/hosp/d_icd_diagnoses/
    # "ROW_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE"
    # old: https://mimic.physionet.org/mimictables/d_icd_diagnoses/
    #
    # subject_id,hadm_id,seq_num,icd_code,icd_version
    # https://mimic-iv.mit.edu/docs/datasets/hosp/diagnoses_icd/
    # "ROW_ID","SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"
    # old: https://mimic.physionet.org/mimictables/diagnoses_icd/
    codes = dataframe_from_csv(os.path.join(mimic4_path, 'd_icd_diagnoses.csv'), index_col=None)
    # there's no SHORT_TITLE in the table, but now there's a version: icd9 or 10
    #codes = codes[['icd9_code','short_title','long_title']]
    codes = codes[['icd_code', 'icd_version', 'long_title']]    # should we limit this data to ICD9?

    diagnoses = dataframe_from_csv(os.path.join(mimic4_path, 'diagnoses_icd.csv'), index_col=None)
    diagnoses = diagnoses.merge(codes, how='inner', left_on=['icd_code', 'icd_version'], right_on=['icd_code', 'icd_version'])
    diagnoses[['subject_id','hadm_id','seq_num']] = diagnoses[['subject_id','hadm_id','seq_num']].astype(int)
    return diagnoses
#=======================================


def read_icd_procedures_table(mimic4_path):
    # icd_code,icd_version,long_title
    # https://mimic-iv.mit.edu/docs/datasets/hosp/d_icd_procedures/
    # "ROW_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE"
    # old: https://mimic.physionet.org/mimictables/d_icd_procedures/
    #
    # subject_id,hadm_id,seq_num,icd_code,icd_version
    # https://mimic-iv.mit.edu/docs/datasets/hosp/procedures_icd/
    # "ROW_ID","SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"
    # old: https://mimic.physionet.org/mimictables/procedures_icd/
    codes = dataframe_from_csv(os.path.join(mimic4_path, 'd_icd_procedures.csv'), index_col=None)
    #codes = codes[['ICD9_CODE','SHORT_TITLE','LONG_TITLE']]
    codes = codes[['icd_code', 'icd_version', 'long_title']]    # should we limit this data to ICD9?

    diagnoses = dataframe_from_csv(os.path.join(mimic4_path, 'procedures_icd.csv'), index_col=None)
    diagnoses = diagnoses.merge(codes, how='inner', left_on=['icd_code', 'icd_version'], right_on=['icd_code', 'icd_version'])
    diagnoses[['subject_id','hadm_id','seq_num']] = diagnoses[['subject_id','hadm_id','seq_num']].astype(int)
    return diagnoses

def read_prescriptions_table(mimic4_path):
    # subject_id,hadm_id,pharmacy_id,starttime,stoptime,drug_type,drug,gsn,ndc,prod_strength,form_rx,dose_val_rx,dose_unit_rx,form_val_disp,form_unit_disp,doses_per_24_hrs,route
    # https://mimic-iv.mit.edu/docs/datasets/hosp/prescriptions/
    # "ROW_ID","SUBJECT_ID","HADM_ID","ICUSTAY_ID","STARTDATE","ENDDATE","DRUG_TYPE","DRUG","DRUG_NAME_POE","DRUG_NAME_GENERIC","FORMULARY_DRUG_CD","GSN","NDC","PROD_STRENGTH","DOSE_VAL_RX","DOSE_UNIT_RX","FORM_VAL_DISP","FORM_UNIT_DISP","ROUTE"
    # old: https://mimic.physionet.org/mimictables/prescriptions/
    prescriptions = dataframe_from_csv(os.path.join(mimic4_path, 'prescriptions.csv'), index_col=None)
    prescriptions.starttime = pd.to_datetime(prescriptions.starttime)   # column labels replaced?
    prescriptions.stoptime = pd.to_datetime(prescriptions.stoptime)      # column labels replaced?

    # ICUSTAY_ID not present in the MIMIC-IV table, so we can't filter here
    ##prescriptions=prescriptions.loc[prescriptions.ICUSTAY_ID.notnull()]
    ##prescriptions['ICUSTAY_ID'] = prescriptions['ICUSTAY_ID'].astype(int)

    prescriptions = prescriptions.loc[prescriptions.ndc != 0]

    #prescriptions=prescriptions.ICUSTAY_ID.notnull()&(prescriptions.ndc!=0)

    prescriptions = prescriptions[
        ['subject_id', 'hadm_id', 'ndc', 'dose_val_rx', 'dose_unit_rx', 'starttime', 'stoptime']]

    #exclude = ['GSN']
    #prescriptions=prescriptions.ix[:, prescriptions.columns.difference(exclude)].hist()
    #print (prescriptions)
    return prescriptions

def merge_on_subject_admission_icustay(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id', 'hadm_id', 'icustay_id'], right_on=['subject_id', 'hadm_id', 'icustay_id'])

#=======================================
def read_events_table_by_row(mimic4_path, table):
    # what is the following dictionary for?
    # looks like the number of rows in each table
    #nb_rows = { 'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219, 'inputevents_cv':17527936, 'inputevents_mv':3618992}
    nb_rows = {'chartevents': 327363275, 'labevents': 122289829, 'outputevents': 4248829,'inputevents_mv': 8869716}
    nb_folder = {'chartevents': 'icu', 'labevents': 'hosp', 'outputevents': 'icu','inputevents_mv': 'icu'}

    reader = csv.DictReader(open(os.path.join(mimic4_path, nb_folder[table.lower()], table.lower() + '.csv'), 'r'))
    for i,row in enumerate(reader):
        if 'stay_id' not in row:
            row['stay_id'] = ''
        yield row, i, nb_rows[table.lower()]

def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[['icd_code','icd_version','long_title']].drop_duplicates().set_index('icd_code')
    counts=diagnoses[['icd_code','hadm_id']].drop_duplicates()
    codes['count1'] = counts.groupby('icd_code')['hadm_id'].count()
    codes.count1 = codes.count1.fillna(0).astype(int)
    codes = codes.ix[codes.count1>0]
    if output_path:
        codes.to_csv(output_path, index_label='icd_code')
    return codes.sort_values('count1', ascending=False).reset_index()

def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id'], right_on=['subject_id'])

def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

def add_age_to_icustays(stays):
    # TODO
    # this function handles transfers as well as stays, and only stays have the anchor_age column
    #
    # we already have the anchor_age in the patients table, so we can use this age
    # we do not have the dob column available
    #stays['age'] = (stays.intime - stays.dob).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    stays['age'] = stays['anchor_age']
    #stays.ix[stays.age<0,'age'] = 90    # this could be omitted if we use anchor_age from patients table
    return stays

def add_inhospital_mortality_to_icustays(stays):
    # will have to do some adoptions here
    # TODO
    mortality = stays.deathtime.notnull() & ((stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime))
    mortality = mortality | (stays.deathtime.isnull() & stays.dod.notnull() & ((stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod)))
    stays['mortality'] = mortality.astype(int)
    stays['mortality_inhospital'] = stays['mortality']
    return stays

def add_inunit_mortality_to_icustays(stays):
    # TODO
    mortality = stays.deathtime.notnull() & ((stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime))
    mortality = mortality | (stays.deathtime.isnull() & stays.dod.notnull() & ((stays.intime <= stays.dod) & (stays.outtime >= stays.dod)))

    stays['mortality_inunit'] = mortality.astype(int)
    return stays

def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays.ix[(stays.age>=min_age)&(stays.age<=max_age)]
    return stays

def filter_diagnoses_on_stays(diagnoses, stays):
    # TODO
    # we do not have any ICUSTAY_ID available, so we can possibly omit it in this merge
    return diagnoses.merge(stays[['subject_id', 'hadm_id', 'stay_id']], how='inner',
                           left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

def break_up_stays_by_subject(stays, output_path, subjects=None, verbose=1):
    # TODO
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays.ix[stays.subject_id == subject_id].sort_values(by='intime').to_csv(os.path.join(dn, 'stays.csv'), index=False)

    if verbose:
        sys.stdout.write('DONE!\n')

def break_up_transfers_by_subject(transfers, output_path, subjects=None, verbose=1):
    # TODO
    subjects = transfers.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        transfers.ix[transfers.subject_id == subject_id].sort_values(by='intime').to_csv(os.path.join(dn, 'transfers.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')

def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None, verbose=1):
    # TODO
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses.ix[diagnoses.subject_id == subject_id].sort_values(by=['stay_id','seq_num']).to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')

def break_up_procedures_by_subject(procedures, output_path, subjects=None, verbose=1):
    # TODO
    subjects = procedures.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i + 1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        procedures.ix[procedures.subject_id == subject_id].sort_values(by=['stay_id', 'seq_num']).to_csv(os.path.join(dn, 'procedures.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


#=======================================

def break_up_prescriptions_by_subject(prescriptions, output_path, subjects=None, verbose=1):
    # TODO
    subjects = prescriptions.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i + 1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        prescriptions.ix[prescriptions.subject_id == subject_id].sort_values(by='starttime').to_csv(os.path.join(dn, 'prescriptions.csv'), index=False)

    if verbose:
        sys.stdout.write('DONE!\n')


# =======================================

def read_events_table_and_break_up_by_subject(mimic4_path, table, output_path, items_to_keep=None, subjects_to_keep=None, verbose=1):
    # TODO
    obs_header = [ 'subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valueuom' ]
    if items_to_keep is not None:
        items_to_keep = set([ str(s) for s in items_to_keep ])
    if subjects_to_keep is not None:
        subjects_to_keep = set([ str(s) for s in subjects_to_keep ])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.last_write_no = 0
            self.last_write_nb_rows = 0
            self.last_write_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        data_stats.last_write_no += 1
        data_stats.last_write_nb_rows = len(data_stats.curr_obs)
        data_stats.last_write_subject_id = data_stats.curr_subject_id
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []
    
    for row, row_no, nb_rows in read_events_table_by_row(mimic4_path, table):
        if verbose and (row_no % 100000 == 0):
            if data_stats.last_write_no != '':
                sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...last write '
                                 '({3}) {4} rows for subject {5}'.format(table, row_no, nb_rows,
                                                                         data_stats.last_write_no,
                                                                         data_stats.last_write_nb_rows,
                                                                         data_stats.last_write_subject_id))
            else:
                sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...'.format(table, row_no, nb_rows))
        
        if (subjects_to_keep is not None and row['subject_id'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None and row['itemid'] not in items_to_keep):
            continue
        
        row_out = { 'subject_id': row['subject_id'],
                    'hadm_id': row['hadm_id'],
                    'stay_id': '' if 'stay_id' not in row else row['stay_id'],
                    'charttime': row['charttime'],
                    'itemid': row['itemid'],
                    'value': row['value'],
                    'valueuom': row['valueuom'] }
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['subject_id']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['subject_id']
        
    if data_stats.curr_subject_id != '':
        write_current_observations()

    if verbose:
        sys.stdout.write('\rfinished processing {0}: ROW {1} of {2}...last write '
                         '({3}) {4} rows for subject {5}...DONE!\n'.format(table, row_no, nb_rows,
                                                                 data_stats.last_write_no,
                                                                 data_stats.last_write_nb_rows,
                                                                 data_stats.last_write_subject_id))
