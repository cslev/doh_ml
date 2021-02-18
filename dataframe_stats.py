#!/usr/bin/python3


import argparse #for argument parsing
import argcomplete #for BASH autocompletion
import pandas as pd #for pandas
import sys # for writing in the same line via sys.stdout
import os

#import own logger function
from logger import Logger
SELF="dataframe_stats.py"


# for histogram generation
import numpy as np
import matplotlib
from matplotlib import pyplot
# Force matplotlib to not use any Xwindows backend - otherwise cannot run script in screen, for instance.
matplotlib.use('Agg')
###


parser = argparse.ArgumentParser(description="Brief analysis of dataframes " +
                                "created via 'create_dataframe.py'" +
                                ".csv files generated via doh_docker container", 
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-i',
                    '--input',
                    action="store",
                    # default="./data_raw/cloudflare/*",
                    type=str,
                    required=True,
                    dest="dataframe_path",
                    help="Specify here the path to the dataframe")

parser.add_argument('-H',
                    '--generate-histogram',
                    action="store_true",
                    default=False,
                    dest="generate_histogram",
                    help="[TRAINING/TESTING] Specify whether to generate " +
                    "histograms for the datasets used for training (Default: " +
                    "False)")

parser.add_argument('-f',
                    '--features',
                    type=str,
                    dest='features',
                    required=False,
                    metavar='N',
                    nargs='+',
                    default=['pkt_len','prev_pkt_len','time_lag','prev_time_lag'],
                    help="Specify the list of features to describe. \n" +
                    "Default: pkt_len, prev_pkt_len, time_lag, prev_time_lag")

#for BASH autocomplete  
argcomplete.autocomplete(parser)

args=parser.parse_args()
DATAFRAME_PATH=args.dataframe_path
DATAFRAME_NAME=os.path.basename(DATAFRAME_PATH).split(".")[0]
# DO WE WANT HISTOGRAM?
HISTOGRAM=args.generate_histogram
HISTOGRAM_PATH="histograms/"

#instantiate logger class
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



def load_datafrom_from_file(path):
  try:
    logger.log(str("Loading from file {}...".format(path)))
    df=pd.read_pickle(path)
    logger.log(str("Loading from file {}...".format(path)),logger.OK)
    return df
  except FileNotFoundError:
    logger.log(str("Loading from file {}...".format(path)), logger.FAIL)
    logger.log_simple("Dataframe {} not found!".format(path))
    logger.log_simple("Try using create_dataframe.py first")
    logger.log_simple("EXITING...")
    exit(-1)


def generate_histogram(dataframe, bin_start, bin_stop, bin_step, column_name, filename_base, xlabel="Packet value", ylabel="Number of packets"):
  logger.log(str("Generating histogram {}...".format(column_name)))
    

  bin_values=np.arange(start=bin_start,stop=bin_stop,step=bin_step)
  doh = dataframe[dataframe['Label']==1]

  #customize tabbing to prettify output
  tabs="\t"
  len_cn=len(column_name)
  if(len_cn < 8):
    tabs=3*tabs
  elif(len_cn > 7 and len_cn < 16):
    tabs=2*tabs
  else: #even longer column name
    tabs=tabs
  try:
      doh = doh[column_name]
  except:
      logger.log(str("Generating histogram {}...".format(column_name)),logger.FAIL)
      logger.log_simple("There was an error during creating the histogram for feature {}".format(column_name))
      logger.log_simple("Probably missing from the dataframe...SKIPPING")
      return
  # print(doh)
  web = dataframe[dataframe['Label']==0]
  web = web[column_name]

  pyplot.clf()
  pyplot.figure()
  pyplot.hist(x=web, bins=bin_values, alpha=0.5, label="Web")
  pyplot.hist(x=doh, bins=bin_values, alpha=0.5, label="DoH")
  pyplot.legend(loc='upper right')

  # fig.subtitle('test title', fontsize=20)
  pyplot.xlabel(xlabel)
  pyplot.ylabel(ylabel)
  pyplot.savefig(HISTOGRAM_PATH+filename_base+".pdf")
  pyplot.savefig(HISTOGRAM_PATH+filename_base+".png")
  logger.log(str("Generating histogram {}...".format(column_name)),logger.OK)
  


#loading dataframe
dataframe = load_datafrom_from_file(DATAFRAME_PATH)

logger.log_simple("COLUMNS", logger.TITLE_OPEN)
logger.log_simple(dataframe.columns)
logger.log_simple("END COLUMNS", logger.TITLE_CLOSE)

logger.log_simple("STATISTICS", logger.TITLE_OPEN)
sum = len(dataframe)
doh = len(dataframe[dataframe["Label"]==1])
web = len(dataframe[dataframe["Label"]==0])
logger.log_simple("Number of packets: {}".format(sum))
logger.log_simple("Number of DoH packets: {} ({:.2f}%)".format(doh, doh/sum*100))
logger.log_simple("Number of Web packets: {} ({:.2f}%)".format(web, web/sum*100))
logger.log_simple("Describing packets", logger.TITLE_OPEN)
for f in FEATURES:
  try:
    logger.log_simple("{}".format(dataframe[[f]].describe()))
  except:
    logger.log_simple("Feature {} is not present in the dataset...SKIPPING".format(f))

logger.log_simple("End Describing packets", logger.TITLE_CLOSE)
logger.log_simple("END STATISITCS", logger.TITLE_CLOSE)

if(HISTOGRAM):
  logger.log_simple("HISTOGRAMS", logger.TITLE_OPEN)
  #check if directory exists
  if not os.path.isdir(HISTOGRAM_PATH):
    logger.log(str("Creating histogram directory {}...".format(HISTOGRAM_PATH)))
    
    try:
      os.mkdir(HISTOGRAM_PATH)
      logger.log(str("Creating histogram directory {}...".format(HISTOGRAM_PATH),logger.OK))
    except:
      logger.log(str("Creating histogram directory {}...".format(HISTOGRAM_PATH),logger.FAIL))
      logger.log_simple(str("Could not create histogram directory {}".format(HISTOGRAM_PATH)))
    
  generate_histogram(dataframe, 
                      50, 
                      300, 
                      1, 
                      "pkt_len", 
                      "histogram_pkt_len_"+DATAFRAME_NAME, 
                      xlabel="Packet Size [B]")
  generate_histogram(dataframe, 
                      50, 
                      300, 
                      1, 
                      "prev_pkt_len", 
                      "histogram_prev_pkt_len_"+DATAFRAME_NAME, 
                      xlabel="Packet Size [B]")
  generate_histogram(dataframe, 
                      0, 
                      0.0005, 
                      0.000001, 
                      "time_lag", 
                      "histogram_time_lag_"+DATAFRAME_NAME, 
                      "Time lag [s]")
  generate_histogram(dataframe, 
                      0, 
                      0.0005, 
                      0.000001, 
                      "prev_pkt_time_lag", 
                      "histogram_prev_time_lag_"+DATAFRAME_NAME, 
                      "Time lag [s]")

  logger.log_simple("END HISTOGRAMS", logger.TITLE_CLOSE)
