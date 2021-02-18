import os
import argparse
import argcomplete #for BASH autocompletion

from logger import Logger #for custom logging


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.model_selection import KFold #For k-fold cross validation

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc



# import joblib
import pickle
import random
import math


#SHAP values
import shap
import warnings

parser = argparse.ArgumentParser(description="This script is for training a " + 
                                "model from DoH traces and test it right away " + 
                                "in a closed-world setting (with optional k-fold "+
                                "cross-validation).\n"+
                                "Use further arguments to obtain more metrics.")

parser.add_argument('-t',
                    '--train-dataframe',
                    action="store",
                    type=str,
                    dest="train_dataframe",
                    required=True,
                    help="Specify the full path for the dataframe "+
                    "used for training")

parser.add_argument('-m',
                    '--ml-model-path',
                    action="store",
                    type=str,
                    dest="ml_model_path",
                    required=True,
                    help="Specify the full path for the model to save")

parser.add_argument('-f',
                    '--features',
                    type=str,
                    dest='features',
                    required=False,
                    metavar='N',
                    nargs='+',
                    default=['pkt_len','prev_pkt_len','time_lag','prev_time_lag'],
                    help="Specify the list of features considered for training (X). " +
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

parser.add_argument('-c',
                    '--cpu-core-num',
                    action="store",
                    default=1,
                    type=int,
                    dest="cpu_core_num",
                    help="Specify here the number of CPU cores to use for " +
                    "parallel jobs (Default: 1)")

parser.add_argument('-C',
                    '--cross-validation',
                    action="store",
                    default=None,
                    type=int,
                    dest="cross_validation",
                    help="Specify here if K-fold cross-validation is needed " +
                    "with the number of fold you want to use (Default: No " + 
                    "cross validation")


#for BASH autocomplete  
argcomplete.autocomplete(parser)
args=parser.parse_args()

SELF="train_model.py"
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


TRAIN_DATAFRAME = args.train_dataframe
ML_MODEL_PATH = args.ml_model_path
# DO WE WANT SHAPLEY VALUES?
SHAPLEY=args.generate_shapley

# DO WE WANT PRC CURVES?
PRC=args.generate_prc

# CPU cores to use
CPU_CORES=args.cpu_core_num

# DO WE WANT CROSS-VALIDATION
CROSS_VALIDATION = args.cross_validation


logger.log_simple("INPUT DATA",logger.TITLE_OPEN)
logger.log("Loading for dataframe {}...".format(TRAIN_DATAFRAME))
if(os.path.isfile(TRAIN_DATAFRAME)):
  try:
    dataframe=pd.read_pickle(TRAIN_DATAFRAME)
    logger.log("Loading for dataframe {}...".format(TRAIN_DATAFRAME), 
                logger.OK)
  except:
    logger.log("Loading for dataframe {}...".format(TRAIN_DATAFRAME), 
                logger.FAIL)
    logger.log_simple("Something happened during reading dataframe")
    logger.log_simple("Exiting...")
    exit(-1)
else:
  logger.log("Loading for dataframe {}...".format(TRAIN_DATAFRAME), 
              logger.FAIL)
  logger.log_simple("File not found")
  logger.log_simple("Exiting...")
  exit(-1)

logger.log_simple("END INPUT DATA",logger.TITLE_CLOSE)




#extracting relevant data from the dataframe
logger.log_simple("Features used for training: {}".format(FEATURES))
# X = dataframe[['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_time_lag']]
X = dataframe[FEATURES]
y = dataframe["Label"]


# initialize classifier
rfc = RandomForestClassifier(n_estimators = 300, 
                              criterion="entropy", 
                              verbose=2, 
                              n_jobs=CPU_CORES)

                              
### CHECK IF Kfold CROSS VALIDATION IS NEEDED
if(CROSS_VALIDATION is not None):
  scores = list()
  cv = KFold(n_splits=CROSS_VALIDATION, random_state=42, shuffle=False)
  logger.log_simple("CROSS VALIDATION",logger.TITLE_OPEN)
  results = dict()
  fold=1 #keep track of folds
  for train_index, test_index in cv.split(X):
    logger.log_simple("Train Index: {}".format(train_index))
    # print(np.take(X,train_index))
    logger.log_simple("Test Index:  {}".format(test_index))
    # print(np.take(X,test_index))
    
    
    X_train = X.iloc[train_index]
    X_test  = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test  = y.iloc[test_index]
    rfc.fit(X_train, y_train)

    rfc_pred = rfc.predict(X_test)
    results[fold] = dict()
    results[fold]["accuracy"]  = metrics.accuracy_score(y_test, rfc_pred)
    results[fold]["precision"] = metrics.precision_score(y_test,rfc_pred)
    results[fold]["recall"]    = metrics.recall_score(y_test, rfc_pred)
    results[fold]["f1-score"]  = metrics.f1_score(y_test,rfc_pred)
    results[fold]["c_matrix"]  = metrics.confusion_matrix(y_test, rfc_pred)
    results[fold]["score"]     = rfc.score(X_test, y_test)

    fold+=1 #keep track of folds
  for f in results:
    logger.log_simple("Fold {}".format(f))
    logger.log_simple(" --> Accuracy:  {}".format(results[f]["accuracy"]))
    logger.log_simple(" --> Precision: {}".format(results[f]["precision"]))
    logger.log_simple(" --> Recall:    {}".format(results[f]["recall"]))
    logger.log_simple(" --> F1-score:  {}".format(results[f]["f1-score"]))
    logger.log_simple(" --> Score:  {}".format(results[f]["score"]))
    logger.log_simple(" --> Confusion Matrix:\n{}".format(results[f]["c_matrix"]))
  
  logger.log_simple("Again, the features used for training: {}".format(FEATURES))
  logger.log_simple("END CROSS VALIDATION",logger.TITLE_CLOSE)


else:
  #train-test splitting (90-10)
  X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size=0.1, 
                                                      random_state=109)


  rfc.fit(X_train,y_train)


  
  logger.log_simple("CLOSED WORLD SETTING",logger.TITLE_OPEN)
  rfc_pred = rfc.predict(X_test)
  logger.log_simple("Accuracy : {}".format(metrics.accuracy_score(y_test, rfc_pred)))
  logger.log_simple("Precision: {}".format(metrics.precision_score(y_test,rfc_pred)))
  logger.log_simple("Recall:    {}".format(metrics.recall_score(y_test, rfc_pred)))
  logger.log_simple("F1 Score:  {}".format(metrics.f1_score(y_test,rfc_pred)))
  logger.log_simple("Confusion Matrix :\n{}".format(metrics.confusion_matrix(y_test, rfc_pred)))
  logger.log_simple("Again, the features used for training: {}".format(FEATURES))

logger.log_simple("END CLOSED WORLD SETTING",logger.TITLE_CLOSE)

# SAVING MODEL
logger.log("Saving the model to {}".format(ML_MODEL_PATH))
try:
  # joblib.dump(rfc, ML_MODEL_PATH, compress=('xz',3)) #compress with xz with compression level 3
  ml_model_file = open(ML_MODEL_PATH, 'wb')
  pickle.dump(rfc, ml_model_file)
  logger.log("Saving the model to {}".format(ML_MODEL_PATH), logger.OK)
except OSError as e:
  logger.log("Saving the model to {}".format(ML_MODEL_PATH), logger.FAIL)
  print(e)
  try:
    logger.log("Removing any residual...")
    os.remove(ML_MODEL_PATH) #remove file is a portion of it has been saved
    logger.log("Removing any residual...", logger.OK)
  except:
    logger.log("Removing any residual...",logger.FAIL)
    logger.log_simple("Could not remove the half-ready model! Delete manually!")
