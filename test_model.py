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


#SHAP values
import shap
import warnings

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
                    required=True,
                    help="Specify output basename used for PRC, shapley, etc.")   

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
ML_MODEL_PATH = args.ml_model_path
SHAPLEY = args.generate_shapley
PRC = args.generate_prc
ROC_AUC = args.generate_roc_auc
CPU_CORES = args.cpu_core_num
OUTPUT = args.output

def load_source(path, label):
  '''
  Load any pickle object and return it.
  path str - the path to the file
  label str - for prettifying output and make output messages meaningful
  '''
  logger.log("Loading for {} {}...".format(label, path))
  if(os.path.isfile(path)):
    try:
      if label == "model":
        retval=joblib.load(path)
      else:
        retval=pd.read_pickle(path)
      logger.log("Loading for {} {}...".format(label,path), 
                  logger.OK)
    except:
      logger.log("Loading for {} {}...".format(label,path), 
                  logger.FAIL)
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
  logger.log_simple("Again, the features used for training: {}".format(FEATURES))

  logger.log_simple("OPEN-WORLD SETTINGS", logger.TITLE_CLOSE)

def make_shap(model, dataframe, basename):
  '''
  Create shapley values for the model given.
  model Model - the model itself
  dataframe Dataframe - the data to work on
  basename str - for storing the plots
  '''

  X = dataframe[FEATURES]
  y = dataframe["Label"]

  explainer = shap.TreeExplainer(model)
  select = range(5)
  pyplot.clf()
  features = X.iloc[select]
  # print(features)
  # features_display = X.loc[features.index]
  # print(features_display)
  # labels = y.iloc[select]
  # print(labels)
  with warnings.catch_warnings() :
      warnings.simplefilter("ignore")
      shap_values = explainer.shap_values(features)[1]
      shap_interaction_values = explainer.shap_interaction_values(features)
  # shap_values = explainer.shap_values(x_train)
  # #shap.force_plot(explainer.expected_value, shap_values[0], features=x_train.loc[0,:], feature_names=x_train.columns)
  if isinstance(shap_interaction_values, list):
      shap_interaction_values = shap_interaction_values[1]
  shap.summary_plot(shap_values, features, plot_type="bar", show=False)
 
  pyplot.savefig(basename+".shapley.pdf")
  pyplot.savefig(basename+".shapley.png")

def generate_roc_auc_csv(model, dataframe, basename, max_fpr=None):
  '''
  Generate ROC AUC curve
  model Model - the model itself
  dataframe Dataframe - the data used for testing as a dataframe
  basename Str - for storing the plots
  max_fpr float - define max_fpr for the ROC AUC curve 
  '''
  X = dataframe[FEATURES]
  y = dataframe["Label"]

  lr_probs = model.predict_proba(X)
  if max_fpr is None:
      print("ROC score (partial AUC): {}".format(roc_auc_score(y, lr_probs[:,1],max_fpr=max_fpr)))
  else:
      print("ROC score (for max FPR {}): {}".format(max_fpr, roc_auc_score(y, lr_probs[:,1],max_fpr=max_fpr)))

  fpr, tpr, _ = roc_curve(y, lr_probs[:,1])

  basename+="ROC_AUC_"
  csv_file  = basename + ".csv"
  plot_file = basename + ".pdf"
  plot_file_png = basename + ".png"
  plot_label = "" #define here any label to add to the plot


  #CSV file
  roc_df = pd.DataFrame({'fpr' : fpr, 'tpr':tpr})
  roc_df.to_csv(csv_file, index_label='index')

  pyplot.clf()
  fig=pyplot.figure()
  pyplot.plot(fpr,tpr, color='navy', lw=2, linestyle='--', label='ROC AUC')
  pyplot.xlim([0.0, 1.0])
  pyplot.ylim([0.0, 1.05])
  pyplot.xlabel('False Positive Rate')
  pyplot.ylabel('True Positive Rate')
  pyplot.title('Receiver operating characteristic')
  pyplot.legend(loc="lower right")
  pyplot.savefig(plot_file)
  pyplot.savefig(plot_file_png)

def generate_pr_csv(model, dataframe, basename):
  """
  Generate precision-recall curve values
  model Model - the model itself
  dataframe Dataframe - the data used for testing as a dataframe
  basename Str - for storing the plots
  """
  X = dataframe[FEATURES]
  y = dataframe["Label"]

  lr_probs = model.predict_proba(X)
  # print("lr_probs:{}".format(lr_probs))
  lr_probs = lr_probs[:, 1]
  # print("lr_probs[:, 1]:{}".format(lr_probs))
  lr_precision, lr_recall, threshold = precision_recall_curve(y, lr_probs)

  basename+="PRC_"
  csv_file  = basename + ".csv"
  plot_file = basename + ".pdf"
  plot_file_png = basename + ".png"
  plot_label = "" #define here any extra label to add to the plot

  # print("Threshold for PRC: {}".format(threshold))
  prc_df = pd.DataFrame({'precision' : lr_precision, 'recall':lr_recall})
  prc_df.to_csv(csv_file, index_label='index')

  pyplot.clf()
  fig=pyplot.figure()
  pyplot.rcParams["figure.figsize"]=5,5
  no_skill = len(y[y==1]) / len(y)
  pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
  pyplot.plot(lr_recall, lr_precision, marker='.', label=plot_label)
  pyplot.xlabel('Recall')
  pyplot.ylabel('Precision')
  pyplot.legend()
  pyplot.savefig(plot_file)
  pyplot.savefig(plot_file_png)


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
  make_shap(rfc, dataframe, OUTPUT)
if PRC:
  generate_pr_csv(rfc, dataframe, OUTPUT)
if ROC_AUC:
  generate_roc_auc_csv(rfc, dataframe, OUTPUT)
