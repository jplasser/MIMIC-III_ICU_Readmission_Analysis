import numpy as np
import os
import pandas as pd

from mimic3benchmark.util import *


def read_stays(subject_path):
    stays = dataframe_from_csv(os.path.join(subject_path, 'stays_readmission.csv'), index_col=None)
    #stays = dataframe_from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    stays.admittime = pd.to_datetime(stays.admittime)
    stays.dischtime = pd.to_datetime(stays.dischtime)
    #stays.DOB = pd.to_datetime(stays.DOB)
    stays.dod = pd.to_datetime(stays.dod)
    stays.deathtime = pd.to_datetime(stays.deathtime)
    stays.sort_values(by=['intime', 'outtime'], inplace=True)
    return stays

def read_diagnoses(subject_path):
    return dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)

def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events.ix[events.value.notnull()]
    events.charttime = pd.to_datetime(events.charttime)
    events.hadm_id = events.hadm_id.fillna(value=-1).astype(int)
    events.stay_id = events.stay_id.fillna(value=-1).astype(int)
    events.valueuom = events.valueuom.fillna('').astype(str)
#    events.sort_values(by=['charttime', 'itemid', 'stay_id'], inplace=True)
    return events
#==============================================
def read_transfers(subject_path):
    stays = dataframe_from_csv(os.path.join(subject_path, 'transfers.csv'), index_col=None)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    #stays.DOB = pd.to_datetime(stays.DOB)
    stays.dod = pd.to_datetime(stays.dod)
    stays.deathtime = pd.to_datetime(stays.deathtime)
    stays.sort_values(by=['intime', 'outtime'], inplace=True)
    return stays

#==============================================
def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = (events.stay_id == icustayid)
    if intime is not None and outtime is not None:
        idx = idx | ((events.charttime >= intime) & (events.charttime <= outtime))
    events = events.ix[idx]
    del events['stay_id']
    return events

def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events['hours'] = (events.charttime - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
    	del events['charttime']
    return events

def convert_events_to_timeseries(events, variable_column='VARIABLE', variables=[]):
    metadata = events[['charttime', 'stay_id']].sort_values(by=['charttime', 'stay_id'])\
                    .drop_duplicates(keep='first').set_index('charttime')
    timeseries = events[['charttime', variable_column, 'value']]\
                    .sort_values(by=['charttime', variable_column, 'value'], axis=0)\
                    .drop_duplicates(subset=['charttime', variable_column], keep='last')
    timeseries = timeseries.pivot(index='charttime', columns=variable_column, values='value').merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries

def get_first_valid_from_timeseries(timeseries, variable):
	if variable in timeseries:
		idx = timeseries[variable].notnull()
		if idx.any():
			loc = np.where(idx)[0][0]
			return timeseries[variable].iloc[loc]
	return np.nan
