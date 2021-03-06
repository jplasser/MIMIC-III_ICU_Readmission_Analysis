import numpy as np
import argparse
import os
os.environ["KERAS_BACKEND"]="tensorflow"

import importlib.machinery
import re
from mimic3benchmark.util import *


from mimic3models.readmission import utils
from mimic3benchmark.readers import ReadmissionReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from utilities.data_loader import get_embeddings
import statistics
import pickle

g_map = { 'F': 1, 'M': 2 }

e_map = { 'ASIAN': 1,
          'BLACK': 2,
          'HISPANIC': 3,
          'WHITE': 4,
          'OTHER': 5, # map everything else to 5 (OTHER)
          'UNABLE TO OBTAIN': 0,
          'PATIENT DECLINED TO ANSWER': 0,
          'UNKNOWN': 0,
          '': 0 }

i_map={'Government': 0,
       'Self Pay': 1,
       'Medicare':2,
       'Private':3,
       'Medicaid':4}


def read_diagnose(subject_path,icustay):
    diagnoses = dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)
    diagnoses=diagnoses.ix[(diagnoses.ICUSTAY_ID==int(icustay))]
    diagnoses=diagnoses['ICD9_CODE'].values.tolist()

    return diagnoses

def get_diseases(names,path):
    disease_list=[]
    namelist=[]
    for element in names:
        x=element.split('_')
        namelist.append((x[0],x[1]))
    for x in namelist:
        subject=x[0]
        icustay=x[1]
        subject_path=os.path.join(path, subject)
        disease = read_diagnose(subject_path,icustay)
        disease_list.append(disease)
    return disease_list

def read_demographic(subject_path,icustay,episode):
    demographic_re=[0]*14
    demographic = dataframe_from_csv(os.path.join(subject_path, episode+'_readmission.csv'), index_col=None)
    age_start=0
    gender_start=1
    enhnicity_strat=3
    insurance_strat=9
    demographic_re[age_start]=float(demographic['Age'].iloc[0])
    demographic_re[gender_start-1+int(demographic['Gender'].iloc[0])]=1
    demographic_re[enhnicity_strat+int(demographic['Ethnicity'].iloc[0])]=1
    insurance =dataframe_from_csv(os.path.join(subject_path, 'stays_readmission.csv'), index_col=None)

    insurance=insurance.ix[(insurance.ICUSTAY_ID==int(icustay))]

    demographic_re[insurance_strat+i_map[insurance['INSURANCE'].iloc[0]]]=1

    return demographic_re

def get_demographic(names,path):
    demographic_list=[]
    namelist=[]
    for element in names:
        x=element.split('_')
        namelist.append((x[0],x[1], x[2]))
    for x in namelist:
        subject=x[0]
        icustay=x[1]
        episode=x[2]

        subject_path=os.path.join(path, subject)
        demographic = read_demographic(subject_path,icustay,episode)
        demographic_list.append(demographic)
    return demographic_list


def disease_embedding(embeddings, word_indices,diseases_list):
    emb_list=[]
    for diseases in diseases_list:
        emb_period=[0]*300
        skip=0
        for disease in diseases:
            k='IDX_'+str(disease)
            if k not in word_indices.keys():
                skip+=1
                continue
            index=word_indices[k]
            emb_disease=embeddings[index]
            emb_period = [sum(x) for x in zip(emb_period, emb_disease)]
        emb_list.append(emb_period)
    return emb_list
#parser = argparse.ArgumentParser()
#common_utils.add_common_arguments(parser)
#parser.add_argument('--target_repl_coef', type=float, default=0.0)
#args = parser.parse_args()
#print (args)

def age_normalize(demographic, age_means, age_std):
    demographic = np.asmatrix(demographic)

    demographic[:,0] = (demographic[:,0] - age_means) / age_std
    return demographic.tolist()

#if args.small_part:
#    args.save_every = 2**30
small_part = False
target_repl = False #(args.target_repl_coef > 0.0 and args.mode == 'train')

base_path = "/system/user/publicwork/student/plasser/MIMIC-III_ICU_Readmission_Analysis/mimic3-readmission"

path = 'train_data/'

# make sure save directory exists
dirname = os.path.dirname(path)
if not os.path.exists(dirname):
    os.makedirs(dirname)

#Read embedding
embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)

train_reader = ReadmissionReader(dataset_dir=f'{base_path}/readm_data/',
                                 listfile=f'{base_path}/MIMIC-III-clean/0_train_listfile801010.csv')

val_reader = ReadmissionReader(dataset_dir=f'{base_path}/readm_data/',
                               listfile=f'{base_path}/MIMIC-III-clean/0_val_listfile801010.csv')

timestep = 1
discretizer = Discretizer(timestep=float(timestep),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

N=train_reader.get_number_of_examples()
ret = common_utils.read_chunk(train_reader, N)
data = ret["X"]
ts = ret["t"]
labels = ret["y"]
names = ret["name"]
diseases_list=get_diseases(names, f'{base_path}/data/')
diseases_embedding=disease_embedding(embeddings, word_indices,diseases_list)
demographic=get_demographic(names, f'{base_path}/data/')

age_means=sum(demographic[:][0])
age_std=statistics.stdev(demographic[:][0])

print('age_means: ', age_means)
print('age_std: ', age_std)
demographic=age_normalize(demographic, age_means, age_std)

headers_from_csv = "Hours,Alanine aminotransferase,Albumin,Alkaline phosphate,Anion gap,Asparate aminotransferase,Basophils,Bicarbonate,Bilirubin,Blood culture,Blood urea nitrogen,Calcium,Calcium ionized,Capillary refill rate,Chloride,Cholesterol,Creatinine,Diastolic blood pressure,Eosinophils,Fraction inspired oxygen,Glascow coma scale eye opening,Glascow coma scale motor response,Glascow coma scale total,Glascow coma scale verbal response,Glucose,Heart Rate,Height,Hematocrit,Hemoglobin,Lactate,Lactate dehydrogenase,Lactic acid,Lymphocytes,Magnesium,Mean blood pressure,Mean corpuscular hemoglobin,Mean corpuscular hemoglobin concentration,Mean corpuscular volume,Monocytes,Neutrophils,Oxygen saturation,Partial pressure of carbon dioxide,Partial pressure of oxygen,Partial thromboplastin time,Peak inspiratory pressure,Phosphate,Platelets,Positive end-expiratory pressure,Potassium,Prothrombin time,Pupillary response left,Pupillary response right,Pupillary size left,Pupillary size right,Red blood cell count,Respiratory rate,Sodium,Systolic blood pressure,Temperature,Troponin-I,Troponin-T,Urine output,Weight,White blood cell count,pH"
header_list_from_csv = headers_from_csv.split(',')

discretizer_header = discretizer.transform(ret["X"][0], header=header_list_from_csv)[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all

data = [discretizer.transform_end_t_hours(X, header=header_list_from_csv, los=t)[0] for (X, t) in zip(data, ts)]

[normalizer._feed_data(x=X) for X in data]
normalizer._use_params()


# Read data
test_reader = ReadmissionReader(dataset_dir=f'{base_path}/readm_data/',
                                listfile=f'{base_path}/MIMIC-III-clean/0_test_listfile801010.csv')

N = test_reader.get_number_of_examples()
re = common_utils.read_chunk(test_reader, N)

names_t = re["name"]
diseases_list_t = get_diseases(names_t, f'{base_path}/data/')
diseases_embedding_t = disease_embedding(embeddings, word_indices, diseases_list_t)
demographic_t = get_demographic(names_t, f'{base_path}/data/')
demographic_t = age_normalize(demographic_t, age_means, age_std)

ret = utils.load_data(test_reader, discretizer, normalizer, diseases_embedding_t, demographic_t, small_part,
                      return_names=True)

data = ret["data"][0]
labels = ret["data"][1]
names = ret["names"]

print ("==> test data generation")
with open(os.path.join(path, 'test_data'), 'wb') as pickle_file:
    pickle.dump(ret, pickle_file)

