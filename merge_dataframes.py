#!/usr/bin/python3

from logger import Logger #for custom logging

import random  #for getting a random snippet from the dataframe at the end

import argparse #for argument parsing
import argcomplete #for BASH autocompletion

import pandas as pd #for pandas
from os import path #for checking paths, dirs, and looping through the files within a dir
import re # regexp for checking csv files convention
import sys # for writing in the same line via sys.stdout



parser = argparse.ArgumentParser(description="Merge Panda dataframes for doh_ml.py", 
                                formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('-i',
                    '--input',
                    action="store",
                    metavar='N',
                    # default="./data_raw/cloudflare/*",
                    type=str,
                    required=True,
                    nargs='+',
                    dest="dataframes",
                    help="Specify here the paths to the dataframes" +
                    "Example: -m /path/to/df1.pkl /path/to/df2.pkl")


parser.add_argument('-o',
                    '--output',
                    action="store",
                    default="./dataframes/df.pkl",
                    type=str,
                    dest="output",
                    help="Specify the full path for the " +
                    "dataframe to be saved (Default: ./dataframes/df.pkl)")


#for BASH autocomplete  
argcomplete.autocomplete(parser)

args=parser.parse_args()

DATAFRAMES=args.dataframes
OUTPUT=args.output

SELF="merge_dataframes.py"
logger = Logger(SELF)

first_dataframe=True
logger.log_simple("DATAFRAMES TO MERGE", logger.TITLE_OPEN)

for p in DATAFRAMES: #iterate through the file list
  if path.isfile(p): #is it a file?
    logger.log("Merging dataframe {}...".format(p))
    df_tmp = pd.read_pickle(p) #load dataframe
    if(first_dataframe): #firt dataframe read
      dataframe = df_tmp #assign
      first_dataframe = False #change bit 
    else: #additional dataframes
      dataframe = pd.concat([dataframe, df_tmp])
    
    logger.log("Merging dataframe {}...".format(p),logger.OK)
    
logger.log_simple("END DATAFRAMES TO MERGE", logger.TITLE_CLOSE)

logger.log("Writing out dataframe to {}...".format(OUTPUT))
dataframe.to_pickle(OUTPUT)
logger.log("Writing out dataframe to {}...".format(OUTPUT),logger.OK)
