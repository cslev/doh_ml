import os
import argparse
import argcomplete #for BASH autocompletion

from logger import Logger #for custom logging


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn import metrics

# from sklearn.model_selection import KFold #For k-fold cross validation

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
from matplotlib import pyplot
# Force matplotlib to not use any Xwindows backend - otherwise cannot run script in screen, for instance.
matplotlib.use('Agg')

import joblib
import pickle
import random
import math

import sys

import misc as misc


import analysis as analysis

parser = argparse.ArgumentParser(description="This script is for loading a " + 
                                "trained model from a file and test it with a "+
                                "dataframe (passed as argument) in an open-world "+
                                "setting.\n"+
                                "Use further arguments to obtain more metrics.")

parser.add_argument('-m',
                    '--ml-model-path',
                    action="store",
                    type=str,
                    dest="ml_model_path",
                    required=True,
                    help="Specify the full path for the model to load")

parser.add_argument('-t',
                    '--test-dataframe',
                    action="store",
                    type=str,
                    dest="test_dataframe",
                    required=True,
                    help="Specify the full path for the dataframe "+
                    "used for testing")           

parser.add_argument('-o',
                    '--output',
                    action="store",
                    type=str,
                    dest="output",
                    default="output_",
                    help="Specify output dir for PRC, shapley, etc.")   

parser.add_argument('-f',
                    '--features',
                    type=str,
                    dest='features',
                    required=False,
                    metavar='N',
                    nargs='+',
                    default=['pkt_len','prev_pkt_len','time_lag','prev_time_lag'],
                    help="Specify the list of features considered for testing (X). " +
                    "This should not contain the 'Label' (y).\n" +
                    "Default: pkt_len, prev_pkt_len, time_lag, prev_time_lag")

parser.add_argument('-S',
                    '--generate-shapley',
                    action="store_true",
                    default=False,
                    dest="generate_shapley",
                    help="Specify whether to generate SHAPLEY values after " + 
                    "testing (Default: False)")

parser.add_argument('-P',
                    '--generate-prc',
                    action="store_true",
                    default=False,
                    dest="generate_prc",
                    help="Specify whether to generate PRC " +
                    "curves after testing (Default: False)")

parser.add_argument('-A',
                    '--generate-roc-auc',
                    action="store_true",
                    default=False,
                    dest="generate_roc_auc",
                    help="Specify whether to generate ROC " +
                    "curves after testing (Default: False)")

parser.add_argument('-c',
                    '--cpu-core-num',
                    action="store",
                    default=1,
                    type=int,
                    dest="cpu_core_num",
                    help="Specify here the number of CPU cores to use for " +
                    "parallel jobs (Default: 1)")                                       

#for BASH autocomplete  
argcomplete.autocomplete(parser)
args=parser.parse_args()

SELF="test_model.py"
logger = Logger(SELF)                    

#check upfront whether set features are valid
POSSIBLE_FEATURES = ['pkt_len','prev_pkt_len','time_lag','prev_time_lag','prev_pkt_time_lag'] #temporarily added the last element
FEATURES = args.features
logger.log("Checking set features to be valid...")
for f in FEATURES:
  if f not in POSSIBLE_FEATURES:
    logger.log("Checking set features to be valid...",logger.FAIL)
    logger.log_simple("Feature {} is not available...EXITING".format(f))
    exit(-1)
logger.log("Checking set features to be valid...",logger.OK)

TEST_DATAFRAME = args.test_dataframe
BASENAME = os.path.basename(TEST_DATAFRAME).split(".")[0]

ML_MODEL_PATH = args.ml_model_path
SHAPLEY = args.generate_shapley
PRC = args.generate_prc
ROC_AUC = args.generate_roc_auc
CPU_CORES = args.cpu_core_num

OUTPUT = args.output
#create output dir if not exists
misc.directory_creator(OUTPUT)

def load_source(path, label):
  '''
  Load any pickle object and return it.
  path str - the path to the file
  label str - if label=model joblib is used to load model file, 
              otherwise pickle is used to load dataframe
  '''
  
  if(os.path.isfile(path)):
    try:
      logger.log("Loading {} from {}...".format(label, path))
      if label == "model":
        retval=joblib.load(path)
      else:
        retval=pd.read_pickle(path)
      logger.log("Loading {} from {}...".format(label, path),logger.OK)
    except:
      logger.log("Loading {} from {}...".format(label, path), logger.FAIL)
      logger.log_simple("Something happened during reading {}".format(label))
      logger.log_simple(sys.exc_info()[0])
      logger.log_simple("Exiting...")
      exit(-1)
  else:
    logger.log("Loading for {} {}...".format(label,path), 
                logger.FAIL)
    logger.log_simple("File not found")
    logger.log_simple("Exiting...")
    exit(-1)

  return retval

def test_model(model, data):
  '''
  This function will test the given data on a given model
  model Model - the model to test the data on
  data dataframe - dataframe describing the data to test
  '''
  #rename model to rfc as we got used to that
  rfc = model
  
  # X = dataframe[['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_time_lag']]
  X = dataframe[FEATURES]
  y = dataframe["Label"]


  logger.log_simple("OPEN-WORLD SETTINGS",logger.TITLE_OPEN)

  logger.log("Testing model...")
  rfc_pred = rfc.predict(X)
  logger.log("Testing model...",logger.OK)

  logger.log_simple("Accuracy : {}".format(metrics.accuracy_score(y, rfc_pred)))
  logger.log_simple("Precision: {}".format(metrics.precision_score(y,rfc_pred)))
  logger.log_simple("Recall:    {}".format(metrics.recall_score(y, rfc_pred)))
  logger.log_simple("F1 Score:  {}".format(metrics.f1_score(y,rfc_pred)))
  logger.log_simple("Confusion Matrix :\n{}".format(metrics.confusion_matrix(y, rfc_pred)))
  logger.log_simple("Again, the features used for testing: {}".format(FEATURES))

  logger.log_simple("OPEN-WORLD SETTINGS", logger.TITLE_CLOSE)


logger.log_simple("INPUT DATA",logger.TITLE_OPEN)
### LOADING DATAFRAME
dataframe = load_source(TEST_DATAFRAME,"dataframe")
### LOADING MODEL
rfc = load_source(ML_MODEL_PATH, "model")
logger.log_simple("END INPUT DATA",logger.TITLE_CLOSE)

#extracting relevant data from the dataframe
logger.log_simple("Features used for testing: {}".format(FEATURES))


test_model(rfc,dataframe)

if(SHAPLEY):
  analysis.make_shap(rfc, dataframe, OUTPUT+"/"+BASENAME, FEATURES)
if PRC:
  analysis.generate_pr_csv(rfc, dataframe, OUTPUT+"/"+BASENAME, FEATURES)
if ROC_AUC:
  analysis.generate_roc_auc_csv(rfc, dataframe, OUTPUT+"/"+BASENAME, FEATURES)
