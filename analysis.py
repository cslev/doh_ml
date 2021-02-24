from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn import metrics

# from sklearn.model_selection import KFold #For k-fold cross validation

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
from matplotlib import pyplot

from logger import Logger #for custom logging
#SHAP values
import shap
import warnings

import numpy as np
import pandas as pd

SELF="analysis.py"
logger = Logger(SELF)   

def make_shap(model, dataframe, basename, features):
  '''
  Create shapley values for the model given.
  model Model - the model itself
  dataframe Dataframe - the data to work on
  basename str - for storing the plots
  '''

  X = dataframe[features]
  y = dataframe["Label"]

  explainer = shap.TreeExplainer(model)
  select = range(5)
  pyplot.clf()
  shap_features = X.iloc[select]
  # print(shap_features)
  # features_display = X.loc[features.index]
  # print(features_display)
  # labels = y.iloc[select]
  # print(labels)
  with warnings.catch_warnings() :
      warnings.simplefilter("ignore")
      shap_values = explainer.shap_values(shap_features)[1]
      shap_interaction_values = explainer.shap_interaction_values(shap_features)
  # shap_values = explainer.shap_values(x_train)
  # #shap.force_plot(explainer.expected_value, shap_values[0], features=x_train.loc[0,:], feature_names=x_train.columns)
  if isinstance(shap_interaction_values, list):
      shap_interaction_values = shap_interaction_values[1]
  shap.summary_plot(shap_values, shap_features, plot_type="bar", show=False)
 
  pyplot.savefig(basename+".shapley.pdf")
  pyplot.savefig(basename+".shapley.png")


def generate_roc_auc_csv(model, dataframe, basename, features, max_fpr=None):
  '''
  Generate ROC AUC curve
  model Model - the model itself
  dataframe Dataframe - the data used for testing as a dataframe
  basename Str - for storing the plots
  max_fpr float - define max_fpr for the ROC AUC curve 
  '''
  X = dataframe[features]
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

def generate_pr_csv(model, dataframe, basename, features):
  """
  Generate precision-recall curve values
  model Model - the model itself
  dataframe Dataframe - the data used for testing as a dataframe
  basename Str - for storing the plots
  """
  X = dataframe[features]
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