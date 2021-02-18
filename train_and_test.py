import os
import argparse

#for regexp matching in the dataset filenames
import re

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score



import matplotlib
from matplotlib import pyplot
# Force matplotlib to not use any Xwindows backend - otherwise cannot run script in screen, for instance.
matplotlib.use('Agg')


import joblib
import random
import math

#SHAP values
import shap
import warnings


TEST_SIZE=0.1
NUM_TREES=300



LOAD_DATAFRAME_BUILD_FROM_SCRATCH=0
LOAD_DATAFRAME_FROM_FILE=1
LOAD_DATAFRAME_SKIP=2

parser = argparse.ArgumentParser(description="Train and test ML model from " +
                                "DoH traces", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-r',
                    '--raw-traffic-traces',
                    action="store",
                    default="cloudflare",
                    dest="traffic_traces",
                    help="[TRAINING] Specify here the traffic traces located " +
                    "in /mnt/storage/Doh_traces separated by commas for " +
                    "training. In other words, the traffic traces (i.e., " +
                    "their .csv files) defined here will be iteratively read " +
                    "and used for training one model (Default: cloudflare, " +
                    "i.e., only cloudflare data will be used for training)")

parser.add_argument('-R',
                    '--raw-traffic-traces-root',
                    action="store",
                    default="/mnt/storage/DoH_traces",
                    dest="traffic_traces_root",
                    help="[TRAINING] Specify here the root directory, where all " +
                    "traffic traces (.csv files) are, each put under a " +
                    "sub-directory used for -r argument (Default: " +
                    "/mnt/storage/Doh_traces)")


parser.add_argument('-D',
                    '--load-dataframe',
                    action="store_true",
                    default=False,
                    dest="load_dataframe",
                    help="[TRAINING/TESTING]Specify whether dataframe should be " +
                    "loaded from file(Default: False, i.e. generate from raw data)")

parser.add_argument('-d',
                    '--dataframe-path',
                    action="store",
                    default="train_df.pkl",
                    type=str,
                    dest="dataframe_path",
                    help="[TRAINING/TESTING]Specify the full path for the " +
                    "dataframe to load/save (Default: ./train_df.pkl)")


parser.add_argument('-M',
                    '--load-ml-model',
                    action="store_true",
                    default=False,
                    dest="load_ml_model",
                    help="[TRAINING/TESTING] Specify whether the ml-model " +
                    "should be loaded from file (Default:False, i.e., trains " +
                    "a model from scratch)")

parser.add_argument('-m',
                    '--ml-model-path',
                    action="store",
                    default="last_model.pkl",
                    type=str,
                    dest="ml_model_path",
                    help="[TRAINING/TESTING] Specify the full path for the " +
                    "model to load/save (Default: None, i.e., model will " +
                    "not be saved)")

parser.add_argument('-p',
                    '--pad-packet-len',
                    action="store",
                    default=None,
                    type=int,
                    dest="pad_pkt_len",
                    help="[TRAINING/TESTING] Specify whether to pad each DoH " +
                    "packet's pkt_len and how (Default: no padding)\n" +
                    "1: Pad according to RFC 8467, i.e., to the closest " +
                    "multiple of 128 bytes.\n" +
                    "2: Pad with a random number between (1,MTU-actual " +
                    "packet size)\n" +
                    "3: Pad to a random number from the distribution of the " +
                    "Web packets\n" +
                    "4: Pad to a random preceding Web packet's size\n" +
                    "5: Pad a sequence of DoH packets to a random sequence of " +
                    "preceeding Web packets' sizes")

parser.add_argument('-H',
                    '--generate-histogram',
                    action="store_true",
                    default=False,
                    dest="generate_histogram",
                    help="[TRAINING/TESTING] Specify whether to generate " +
                    "histograms for the datasets used for training (Default: " +
                    "False)")
parser.add_argument('-S',
                    '--generate-shapley',
                    action="store_true",
                    default=False,
                    dest="generate_shapley",
                    help="[TRAINING/TESTING] Specify whether to generate " +
                    "SHAPLEY values after testing (Default: False)")

parser.add_argument('-P',
                    '--generate-prc',
                    action="store_true",
                    default=False,
                    dest="generate_prc",
                    help="[TRAINING/TESTING] Specify whether to generate PRC " +
                    "curves after testing (Default: False)")

parser.add_argument('-A',
                    '--generate-roc-auc',
                    action="store_true",
                    default=False,
                    dest="generate_roc_auc",
                    help="[TRAINING/TESTING] Specify whether to generate ROC " +
                    "curves after testing (Default: False)")


parser.add_argument('-f',
                    '--max-fpr',
                    action="store",
                    default=None,
                    type=float,
                    dest="max_fpr",
                    help="[MISC] Specify here the maximum FPR you want the TPR " +
                    "to be printed in case of generating an ROC AUC curve. " +
                    "Use only in conjunction with -R (Default: None)")


parser.add_argument('-t',
                    '--pad-time-lag',
                    action="store",
                    default=None,
                    type=int,
                    dest="pad_time_lag",
                    help="[TRAINING/TESTING] Specify whether to pad each DoH " +
                    "packet's time_lag and how (Default: no padding)\n" +
                    "3: Pad to a random number from the distribution of the " +
                    "Web packets\n" +
                    "4: Pad to a random preceding Web packet's size\n" +
                    "5: Pad a sequence of DoH packets to a random sequence of " +
                    "preceeding Web packets' sizes")


parser.add_argument('-o',
                    '--open-world-dataframe',
                    action="store",
                    default=None,
                    dest="open_world_dataframe",
                    help="[TESTING] Open-world setting: Specify here the path " +
                    "to the data frame you want to use for testing. If no " +
                    "dataframe is avalilable, use -u setting instead to point " +
                    "to the raw files! (Default:None, i.e. Closed-world setting)")

parser.add_argument('-O',
                    '--open-world-raw-data',
                    action="store",
                    default=None,
                    dest="open_world_raw_data",
                    help="[TESTING] Open-world setting: Specify here the path " +
                    "to the raw data to use for testing. In this case, the raw " +
                    "csv files will be used under the directory specified via " +
                    "this parameter to build the data frame first. If " +
                    "dataframe is avalilable, use -o setting instead! " +
                    "(Default: None, i.e. Closed-world setting)")

parser.add_argument('-C',
                    '--create-dataframe-only',
                    action="store_true",
                    default=False,
                    dest="dataframe_only",
                    help="[MISC] Set if dataframe creation and saving is " +
                    "needed only! Use only with -d/--dataframe-path to set " +
                    "where to store the dataframe (Default: False)")



parser.add_argument('-y',
                    '--output-dir',
                    action="store",
                    default="./",
                    dest="output_dir",
                    help="[MISC] Output directory where histograms, shapley " +
                    "values, PRC curves and accuracy metrics will be saved. " +
                    "(Default: .)")

parser.add_argument('-c',
                    '--cpu-core-num',
                    action="store",
                    default=1,
                    type=int,
                    dest="cpu_core_num",
                    help="[MISC] Specify here the number of CPU cores to use " +
                    "for parallel jobs (Default: 1)")

parser.add_argument('-j',
                    '--override-meta',
                    action="store",
                    default=None,
                    type=str,
                    dest="override_meta",
                    help="[MISC] Specify here any extra metadata (Default: None)")


parser.add_argument('-w',
                    '--write-dataframe-to-csv',
                    action="store_true",
                    default=False,
                    dest="write_dataframe_to_csv",
                    help="[MISC] Specify here if dataframe is needed to be printed as CSV (Default: False)")





args=parser.parse_args()
#check which arguments are set and not set that depend on each other (these arguments have either boolean or None set as default value, so the following checks will work. For other arguments, we cannot set default values during addition, we have to set them later.)
if args.open_world_dataframe and args.open_world_raw_data:
    parser.error("If -o/--open-world-dataframe is set then -u/--open-world-raw-data CANNOT be set!")
    exit(-1)

if args.load_dataframe and not args.dataframe_path:
    parser.error("If -d/--load-dataframe is set then -n/--dataframe-path HAS TO be specified!")
    exit(-1)

if args.load_ml_model and not args.ml_model_path:
    parser.error("If -m/--load-ml-model is set then -i/--ml-model-path HAS TO be specified!")
    exit(-1)

if args.dataframe_only and not args.dataframe_path:
    parser.error("If -C/--create-dataframe-only is set, -d/--dataframe-path has to be specified")
    exit(-1)

# Dataframe related
LOAD_DATAFRAME=args.load_dataframe
#LOAD_DATAFRAME=True/False
if(LOAD_DATAFRAME):
    LOAD_DATAFRAME=LOAD_DATAFRAME_FROM_FILE #we load dataframe from file
    #LOAD_DATAFRAME=1
else:
    LOAD_DATAFRAME=LOAD_DATAFRAME_BUILD_FROM_SCRATCH #we build dataframe from scratch
    #LOAD_DATAFRAME=0

DATAFRAME_NAME_WITH_FULL_PATH=args.dataframe_path
DATAFRAME_NAME=os.path.basename(DATAFRAME_NAME_WITH_FULL_PATH) #remove path arguments if there is any
PATH_TO_DATAFRAME=DATAFRAME_NAME_WITH_FULL_PATH.replace(DATAFRAME_NAME,"")

# create dataframe only?
CREATE_DATAFRAME_ONLY=args.dataframe_only

#Model related
LOAD_MODEL=args.load_ml_model
MODEL_NAME_WITH_FULL_PATH=args.ml_model_path
MODEL_NAME=None
if(MODEL_NAME_WITH_FULL_PATH is not None):
    MODEL_NAME=os.path.basename(args.ml_model_path) #remove path arguments if there is any
    PATH_TO_MODEL=MODEL_NAME_WITH_FULL_PATH.replace(MODEL_NAME,"")


#PADDING related
PAD_PKT_LEN=args.pad_pkt_len
PAD_TIME_LAG=args.pad_time_lag

#TRACES
RESOLVERS_FOR_TRAINING=(args.traffic_traces).split(",")
BASE_PATH_TO_DATASETS  = args.traffic_traces_root+"/"

#OPEN WORLD SETTING for DATAFRAME
OPEN_WORLD = args.open_world_dataframe

#OPEN WORLD SETTING for raw data
OPEN_WORLD_RAW_DATA = args.open_world_raw_data
#check if last character is '/'
if OPEN_WORLD_RAW_DATA is not None and OPEN_WORLD_RAW_DATA[-1] == "/":
    #we need to remove it otherwise functions later mess this up
    OPEN_WORLD_RAW_DATA = OPEN_WORLD_RAW_DATA[:-1] #cut the last character

if(OPEN_WORLD is None) and (OPEN_WORLD_RAW_DATA is None):
    CLOSED_WORLD=True
else:
    CLOSED_WORLD=False

    #if it is not close world scenario and we have loaded our model, no need for building any dataframe from scratch for closed-world testing
    if(LOAD_MODEL):
        #open world dataframes are anyway built in this case
        LOAD_DATAFRAME=LOAD_DATAFRAME_SKIP

# DO WE WANT HISTOGRAM?
HISTOGRAM=args.generate_histogram

# DO WE WANT SHAPLEY VALUES?
SHAPLEY=args.generate_shapley

# DO WE WANT PRC CURVES?
PRC=args.generate_prc

# DO WE WANT ROC AUC CURVES?
ROC_AUC=args.generate_roc_auc
MAX_FPR=args.max_fpr

OUTPUT_DIR=args.output_dir

CPU_CORES=args.cpu_core_num

#create metadata name
OVERRIDE_META = args.override_meta
#if (len(RESOLVERS_FOR_TRAINING) == 1):
#    META = str(RESOLVERS_FOR_TRAINING[0])
#else:
#    META = "combined_"+str(len(RESOLVERS_FOR_TRAINING))
META=""
if (CLOSED_WORLD):
    META = META + "_cw"
    print("Training - Testing ratio: {} - {}".format((1-TEST_SIZE), TEST_SIZE))
elif((OPEN_WORLD is not None) and (OPEN_WORLD_RAW_DATA is None)):
    META = META + "_ow_"+os.path.basename(OPEN_WORLD).split(".")[0] #we don't need the trailing file extension
elif((OPEN_WORLD is None) and (OPEN_WORLD_RAW_DATA is not None)):
    META = META + "_ow_"+os.path.basename(OPEN_WORLD_RAW_DATA).split(".")[0] #we don't need the trailing file extension
else:
    pass

if ((PAD_PKT_LEN is None) and (PAD_TIME_LAG is None)):
    META = META + "_nopad_"

elif(PAD_PKT_LEN is not None):
    META = META + "_pad_p_"+str(PAD_PKT_LEN)
elif(PAD_TIME_LAG is not None):
    META = META + "_pad_t_"+str(PAD_TIME_LAG)

if(OVERRIDE_META is not None):
    META = OVERRIDE_META
print("METADATA is set as: {}".format(META))



WRITE_DATAFRAME_TO_CSV = args.write_dataframe_to_csv


#------------------------------- CHECKING PATHS ----------------------------
print("CHECKING GIVEN PATHS...")
if(not os.path.exists(BASE_PATH_TO_DATASETS)):
    print("Base path to datasets {} does not exists!".format(BASE_PATH_TODATASETS))
    print("Exiting...")
    exit(-1)

if(LOAD_DATAFRAME == LOAD_DATAFRAME_FROM_FILE and not os.path.exists(DATAFRAME_NAME_WITH_FULL_PATH)):
    print("Dataframe {} to load cannot be found!".format(DATAFRAME_NAME_WITH_FULL_PATH))
    print("Exiting...")
    exit(-1)
if(LOAD_MODEL and not os.path.exists(MODEL_NAME_WITH_FULL_PATH)):
    print("Model {} to load cannot be found!".format(MODEL_NAME_WITH_FULL_PATH))
    print("Exiting...")
    exit(-1)
if(CLOSED_WORLD == False):
    if(OPEN_WORLD_RAW_DATA is not None):
        if(not os.path.exists(OPEN_WORLD_RAW_DATA)):
            print("Path to open-world raw data {} does not exists!".format(OPEN_WORLD_RAW_DATA))
            print("Exiting...")
            exit(-1)
    if(OPEN_WORLD is not None):
        if(not os.path.exists(OPEN_WORLD)):
            print("Path to open-world dataframe {} does not exists!".format(OPEN_WORLD))
            print("Exiting...")
            exit(-1)
print("\t[DONE]")

#===========================================================================

#################################
# ROGRAM DEFINITIONS START HERE #
#################################
#required only for RFC based padding for pkt_len values (if PAD_PKT_LEN=1)
RFC_PADDING=dict()
for i in range(1,12):
    RFC_PADDING[i]=i*128
RFC_PADDING[12] = 1500

# further variables required for the different padding techniques
iterator = 0 # for padding techniques 4 and 5
prev_http_pkt_len = 0 # for padding techniques 4 and 5
prev_http_time_lag = 0.0 # for padding techniques 4 and 5
is_http_new = True # for padding techniques  5
prev_http_packets = [] # for padding techniques 4 and 5
# for padding techniques 4 and 5
WEBPACKET_BUFFER_SIZE_PKTLEN = 30 #empirically the best among 5,10,20,30
WEBPACKET_BUFFER_SIZE_TIMELAG = 30 #empirically the best among 5,10,20,30


#many doh resolvers have /dns-query as a last argument of the POST query
#CleanBrowsing has /doh/family-filter/
#powerDNS has / only
HTTP_POST_QUERIES_TO_RELABEL=["/dns-query", "/doh/family-filter/", "/"]



def add_doh_resolver(df,resolver) :
    l = len(df)
    res = []
    for i in range(l) :
        res.append(resolver)
    df['resolver'] = res
    return df

def add_packet_direction(df_temp) :
    direction= []
    for i in df_temp['Source']:
        # print(str(i))
        if (i[0:12]=='192.168.122.' or #local IP for the 4 resolvers
            i[0:7]=='172.17.' or #docker IPs
            i[0:8]=='128.110.' or #cloudlabs IPs utah
            i[0:8]=='128.105.' or #cloudlabs IPs wisconsin
            i[0:12]=='132.227.122.'): #cloudlabs arm
            direction.append('request')
        else :
            direction.append('response')
    df_temp['direction']=direction
    return df_temp

def remove_non_important(df_temp) :
    idx = []
    for id,value in enumerate(df_temp['Protocol']):
        if value!='HTTP2' and value!='DoH':
            idx.append(id)
    df_temp = df_temp.drop(idx)
    return df_temp

def add_label(df) :
    Label = []
    df_temp=df[['Protocol','Info']]
    for a , b in df_temp.itertuples(index=False) :
        if a == 'DoH' :
            Label.append(int(1))
        else :
            b=str(b)
            if(b.split(" ")[-1] in HTTP_POST_QUERIES_TO_RELABEL) :
                Label.append(int(1))
            else:
                Label.append(int(0))
    df['Label']=Label
    return df

def add_time_lag(df_temp) :
    time_lag = []
    prev = df_temp['Time'].iloc[0]
    for i in df_temp['Time'] :
        lag = i - prev
        time_lag.append(lag)
        prev = i

    df_temp['time_lag'] = time_lag
    return df_temp

def add_previous_packet_lag(df_temp) :
    prev_lag = [0.0,]
    for i in df_temp['time_lag'] :
        prev_lag.append(i)
    prev_lag = prev_lag[:-1]
    df_temp['prev_pkt_time_lag'] = prev_lag
    return df_temp

def add_packet_difference(df_temp) :
    diff = []
    prev = df_temp['No.'].iloc[0]
    for i in df_temp['No.'] :
        diff.append(i-prev)
        prev = i
    df_temp['seq_no_diff'] = diff
    return df_temp

def add_previous_packet_length(df_temp) :
    prev_len = [0,]
    flag = 0
    for i in df_temp['pkt_len'] :
        prev_len.append(int(i))
    prev_len= prev_len[:-1]
    df_temp['prev_pkt_len'] = prev_len
    return df_temp

def get_random_choice_from_list(list_of_values, value_to_be_greater_than):
    new_list = []
    # create a new list of the bigger values only
    for i in list_of_values:
        if i > value_to_be_greater_than:
            new_list.append(i)
    #if there was any element that was bigger than value_to_be_greater_than
    if len(new_list) > 0:
        #get random element from the new list
        retval = random.choice(new_list)
        return retval
    #otherwise, return None that will be handled at the catching side
    else:
        return None

def get_next_element_from_list(list_of_values, value_to_be_greater_than, iterator=None):
    #create a list only from the bigger numbers
    new_list = []
    # create a new list of the bigger values only
    for i in list_of_values:
        if i > value_to_be_greater_than:
            new_list.append(i)
    #if there was any element that was bigger than value_to_be_greater_than
    if len(new_list) > 0:
        if (iterator is None):
            iterator = random.randint(0,len(new_list)-1)
        else:
            iterator = iterator + 1
            if (iterator > len(new_list)-1):
                iterator = random.randint(0,len(new_list)-1)


        #get random element from the new list
        retval = new_list[iterator]
        return retval,iterator
    #otherwise, return None that will be handled at the catching side
    else:
        return None, None

def update_biggest_web_packet_list(pkt_len):
    if len(biggest_web_packets) < BIGGEST_WEB_PACKETS_SIZE:
        #if we have space, we don't care just add it to the list
        biggest_web_packets.append(pkt_len)
    else:
        #check whether the new value is greater than any of the values
        temp=None
        was_bigger = False
        original_pkt_len = pkt_len
        for i,web_packet in enumerate(biggest_web_packets):
            if web_packet < pkt_len:
                temp = i
                pkt_len = web_packet
                was_bigger = True
        if was_bigger:
            del biggest_web_packets[temp]
            biggest_web_packets.append(original_pkt_len)

def pad_pkt_len(pkt_len, Label, mean, stdev):
    global prev_http_pkt_len
    global prev_http_packets
    global is_http_new
    global iterator

    pkt_len=int(pkt_len)
    #DoH PACKET
    if(int(Label) == 1):
        # print(PAD_PKT_LEN, type(PAD_PKT_LEN))
        if(PAD_PKT_LEN == 1):
        # RFC 8467
            if(pkt_len < 1500): # if, by any chance the packet would be higher than 1500 bytes, we don't do anything
                index = int(pkt_len/128) + 1
                #pad each packet by the mean value+-stdev of WEB packets
                new_pkt_len = RFC_PADDING[index]
            else:
                new_pkt_len = pkt_len #pkt_len was bigger than MTU, return it and consider it a measurement error :D

        # Random padding to not exceed MTU
        elif(PAD_PKT_LEN == 2):
            if(pkt_len < 1500):
                upper_bound = 1500-abs(pkt_len)
                # print(upper_bound)
                new_pkt_len = pkt_len + random.randint(1, upper_bound)
            else:
                new_pkt_len = pkt_len


        # Pad to a random size from the normal distribution of the web traffic
        elif(PAD_PKT_LEN == 3):
            if(pkt_len < 1500):
                new_pkt_len = int(np.random.normal(mean,stdev))
                if(new_pkt_len < pkt_len):
                    new_pkt_len = pkt_len + new_pkt_len
            else:
                new_pkt_len = pkt_len


        # pad to a random picked preceding Web packet's size
        elif(PAD_PKT_LEN == 4):
            if len(prev_http_packets) > 2:
                new_pkt_len = get_random_choice_from_list(prev_http_packets, pkt_len)
                if new_pkt_len == None: #if there is no greater element than current pkt_len
                    new_pkt_len = int(np.random.normal(mean,stdev)) #technique 4
                    if(new_pkt_len < pkt_len):
                        new_pkt_len = pkt_len + new_pkt_len
            else:
                new_pkt_len = int(np.random.normal(mean,stdev)) #technique 4
                if(new_pkt_len < pkt_len):
                    new_pkt_len = pkt_len + new_pkt_len

        # pad to a randomly selected sequence of the preceding Web packets' sizes
        elif(PAD_PKT_LEN == 5):
            # iterator = 0
            if len(prev_http_packets) > 3:
                if is_http_new:
                    #first DoH, get a random element from the prev web packets
                    iterator = random.randint(0,len(prev_http_packets)-2)
                    is_http_new = False

                else:
                    # consecutive DoH packets, step ahead by one in the web packet list
                    iterator = iterator + 1
                    if(iterator == len(prev_http_packets)-1):
                        #in case we overflow the list, we grab a new random element from the list
                        iterator = random.randint(0,len(prev_http_packets)-2)

                new_pkt_len = prev_http_packets[iterator]

                #if in any case the picked pkt_len value is smaller than the actual DoH packet's pkt_len value
                if (new_pkt_len < pkt_len):
                    #we jump at a random place in the list of web packets, get the first pkt_len from there, which is greater than the DoH packet's value
                    new_pkt_len_tmp = new_pkt_len
                    for i in range(random.randint(0,len(prev_http_packets)-1), len(prev_http_packets)-1):
                        if(prev_http_packets[i] > pkt_len):
                            new_pkt_len=prev_http_packets[i]
                            #if found, break the loop
                            break
                    if(new_pkt_len == new_pkt_len_tmp): #there was no bigger packet
                        new_pkt_len = int(np.random.normal(mean,stdev)) #we just follow again PAD_PKT_LEN=3 technique
                        if(new_pkt_len < pkt_len):
                            new_pkt_len = pkt_len + new_pkt_len

            #this applies only in the beginning, when there is not enough Web packets from the past
            else:
                new_pkt_len = int(np.random.normal(mean,stdev)) #technique 4
                if(new_pkt_len < pkt_len):
                    new_pkt_len = pkt_len + new_pkt_len

        else:
            print("UNSUPPORTED PADDING TECHNIQUE {}".format(PAD_PKT_LEN))
            exit(-1)

        return new_pkt_len

    # WEB PACKET
    else:
        # Padding technique 4 and 5
        if len(prev_http_packets) > WEBPACKET_BUFFER_SIZE_PKTLEN-1:
            #if we reach WEBPACKET_BUFFER_SIZE, then we constantly remove the first element,
            # before adding the new one to keep the element size to WEBPACKET_BUFFER_SIZE
            # Actually, once this is reached, the condition will always be satisfied
            del prev_http_packets[0]
        prev_http_packets.append(int(pkt_len))

        # for padding technique 5, we also need this indicator variable
        #when a Web packet is received we set this back to True to indicate technique 5 that a new subsequence should be chosen
        #then, it will set this value back to False which remains False unless again a new Web packet is received
        is_http_new = True

        #update biggest web biggest_web_packets
        # update_biggest_web_packet_list(pkt_len)



        # return the original value for Web packets
        return pkt_len

def pad_time_lag(time_lag, Label, mean, stdev):
    global prev_http_pkt_len
    global prev_http_time_lag
    global prev_http_packets
    global is_http_new
    global iterator
    if(int(Label) == 1):
        # print(PAD_PKT_LEN, type(PAD_PKT_LEN))
        # Pad to a random size from the normal distribution of the web traffic
        if(PAD_TIME_LAG == 3):
            new_time_lag = float(np.random.normal(mean,stdev))
            if(new_time_lag < time_lag):
                new_time_lag = time_lag + new_time_lag

        # pad to a random picked preceding Web packet's size
        elif(PAD_TIME_LAG == 4):
            if len(prev_http_packets) > 2:
                new_time_lag = get_random_choice_from_list(prev_http_packets, time_lag)
                if new_time_lag == None: #if there is no greater element than current time_lag
                    new_time_lag = float(np.random.normal(mean,stdev)) #technique 4
                    if(new_time_lag < time_lag):
                        new_time_lag = time_lag + new_time_lag
            else:
                new_time_lag = float(np.random.normal(mean,stdev)) #technique 4
                if(new_time_lag < time_lag):
                    new_time_lag = time_lag + new_time_lag

        # pad to a randomly selected sequence of the preceding Web packets' sizes
        elif(PAD_TIME_LAG == 5):
            if len(prev_http_packets) > 3:
                if is_http_new:
                    #first DoH, get a random element from the prev web packets
                    iterator = random.randint(0,len(prev_http_packets)-2)
                    is_http_new = False

                else:
                    # consecutive DoH packets, step ahead by one in the web packet list
                    iterator = iterator + 1
                    if(iterator == len(prev_http_packets)-1):
                        #in case we overflow the list, we grab a new random element from the list
                        iterator = random.randint(0,len(prev_http_packets)-2)

                new_time_lag = prev_http_packets[iterator]

                #if in any case the picked pkt_len value is smaller than the actual DoH packet's pkt_len value
                if (new_time_lag < time_lag):
                    #we jump at a random place in the list of web packets, get the first pkt_len from there, which is greater than the DoH packet's value
                    new_time_lag_tmp = new_time_lag
                    for i in range(random.randint(0,len(prev_http_packets)-1), len(prev_http_packets)-1):
                        if(prev_http_packets[i] > time_lag):
                            new_time_lag=prev_http_packets[i]
                            #if found, break the loop
                            break
                    if(new_time_lag == new_time_lag_tmp): #there was no bigger packet
                        new_time_lag = float(np.random.normal(mean,stdev)) #we just follow again PAD_PKT_LEN=3 technique
                        if(new_time_lag < time_lag):
                            new_time_lag = time_lag + new_time_lag

            #this applies only in the beginning, when there is not enough Web packets from the past
            else:
                new_time_lag = float(np.random.normal(mean,stdev)) #technique 4
                if(new_time_lag < time_lag):
                    new_time_lag = time_lag + new_time_lag

        else:
            print("UNSUPPORTED PADDING TECHNIQUE {}".format(PAD_TIME_LAG))
            exit(-1)

        return new_time_lag

    else:
        if len(prev_http_packets) > WEBPACKET_BUFFER_SIZE_TIMELAG-1:
            #if we reach WEBPACKET_BUFFER_SIZE, then we constantly remove the first element,
            # before adding the new one to keep the element size to WEBPACKET_BUFFER_SIZE
            # Actually, once this is reached, the condition will always be satisfied
            del prev_http_packets[0]
        prev_http_packets.append(float(time_lag))

        # for padding technique 5, we also need this indicator variable
        #when a Web packet is received we set this back to True to indicate technique 5 that a new subsequence should be chosen
        #then, it will set this value back to False which remains False unless again a new Web packet is received
        is_http_new = True

        return time_lag

def pad_feature_values(df_temp, pkt_len=False, time_lag=False):
    global prev_http_pkt_len
    global prev_http_time_lag
    global is_http_new
    global prev_http_packets
    web = df_temp[df_temp["Label"] == 0]
    if(pkt_len):
        prev_http_pkt_len=0
        is_http_new=True
        prev_http_packets=[]

        pkt_len_mean  = web.pkt_len.mean()
        pkt_len_stdev = web.pkt_len.std()
        print("Padding pkt_len values...")
        print("Statistics:")
        print("WEB pkt_len_mean: {}".format(pkt_len_mean))
        print("WEB pkt_len_stdev: {}".format(pkt_len_stdev))
        # print(df_temp)
        df_temp['pkt_len'] = df_temp.apply(lambda x : pad_pkt_len(x.pkt_len, x.Label, pkt_len_mean, pkt_len_stdev), axis=1)
        # print(df_temp)
        print("Padding pkt_len/prev_pkt_len is DONE")
    if(time_lag):
        prev_http_time_lag=0
        is_http_new=True
        prev_http_packets=[]

        time_lag_mean = web.time_lag.mean()
        time_lag_stdev = web.time_lag.std()
        print("Padding time_lag values...")
        print("Statistics:")
        print("WEB time_lag_mean: {}".format(time_lag_mean))
        print("WEB time_lag_stdev: {}".format(time_lag_stdev))
        df_temp['time_lag'] = df_temp.apply(lambda x : pad_time_lag(x.time_lag, x.Label, time_lag_mean, time_lag_stdev), axis=1)
        # print(df_temp)
        # exit(-1)
        print("Padding time_lag/prev_pkt_time_lag is DONE")
    else:
        print("No padding was done")
#    print("Padding is DONE")

    return df_temp

def create_dataframe(filename,resolver) :
    # print(filename)
    df = pd.read_csv(filename, header=0, names=["No.","Time","Source","Destination","Protocol","pkt_len","Info"])
    # manipulating data
    df = df[1:]
    #append dataframe with resolver column
    df = add_doh_resolver(df,resolver)
    #append dataframe with direction column
    df = add_packet_direction(df)
    #remove non-HTTPS and non-DoH packets
    df = remove_non_important(df)

    #filter only on requests - Scenario 2 finetuning
    df = df[df.direction == "request"]

    #label dataframe
    df = add_label(df)
    # add time_lags
    df = add_time_lag(df)
    # print(df)
    # exit(-1)

    df = pad_feature_values(df, pkt_len=PAD_PKT_LEN, time_lag=PAD_TIME_LAG)

    # print(df)
    df = add_previous_packet_length(df)

    df = add_previous_packet_lag(df)
    df = add_packet_difference(df)

    # df = df[1:]
    # df = df['No.', 'pkt_len', 'prev_pkt_len', 'time_lag', 'prev_pkt_time_lag', 'seq_no_diff', 'direction', 'Protocol', 'Label','Info', 'Source', 'Destination','resolver']
    doh = df[df['Label']==1]
    http = df[df['Label']==0]

    # df = df[['pkt_len', 'prev_pkt_len','Label']]
    # df = df[['pkt_len', 'Label']]

    return df

def make_dataframe( load_dataframe=LOAD_DATAFRAME,
                    dataframe_path=DATAFRAME_NAME_WITH_FULL_PATH,
                    dataframe_name=DATAFRAME_NAME,
                    resolvers=RESOLVERS_FOR_TRAINING,
                    base_path_to_datasets=BASE_PATH_TO_DATASETS):
    '''
    This function is used to create the datapath for training, but also used to create dataframes for testing.
    For training, the constant variables set by command-line arguments are used to locate the dataframe to load, or the
    path where the raw data is.
    This function is called with explicit variable settings in the train_and_test(), where we use this for gathering the testing dataset.
    Similarly, if the testing dataset is not found as a dataframe, we build the dataframe from scratch.
    @params
    load_dataframe Int - indicate explicitly to load a dataframe (0: build from scratch, 1: load, 2: no load and don't build as it is not required)
    dataframe_path String - full path to the dataframe to load
    dataframe_name String - only the name of the dataframe (used for saving and logging)
    resolvers List(String) - list of resolvers we want to generate dataframes for
    base_path_to_datasets String - the base explicit path where the raw data is. All resolvers defined in the resolver list must be located in the directory defined via this variable
    '''
    #for reference
    #LOAD_DATAFRAME_BUILD_FROM_SCRATCH=0
    #LOAD_DATAFRAME_FROM_FILE=1
    #LOAD_DATAFRAME_SKIP=2
    if(load_dataframe == LOAD_DATAFRAME_FROM_FILE):
        try:
            print("\tLoading from file {}...".format(dataframe_path))
            dataframe=pd.read_pickle(dataframe_path)
            #print("[DONE]")
            # print(dataframe)
            return dataframe
        except FileNotFoundError:
            # print("Dataframe {} not found! Rebuilding from scratch...".format(dataframe_name))
            print("\tDataframe {} not found! EXITING...".format(dataframe_name))
            exit(-1)
            #recursively call this same function with branching to the else branch
            # make_dataframe(load_dataframe=False)
    elif(load_dataframe == LOAD_DATAFRAME_BUILD_FROM_SCRATCH):
        print("\tCreating from scratch...")
        dataframe=get_data_for_training(resolvers, base_path_to_datasets)
        print("\tSaving dataframe to {}".format(dataframe_path))
        dataframe.to_pickle(dataframe_path)
        return dataframe
    else: #LOAD_DATAFRAME=LOAD_DATAFRAME_SKIP
        print("\tSKIPPING!")
        return None

def get_data_for_training(list_of_resolvers, base_path):
    tmp_res=dict()
    #print(list_of_resolvers)
    #print(base_path)
    for r in list_of_resolvers:
        tmp_res[r] = list()
        #print(base_path+r)
        if not os.path.exists(base_path+r):
            print("Path {} does not exists...exiting".format(str(base_path+r)))
            exit(-1)

        for _,_,files in os.walk(base_path+r):
            #print(files)
            for filename in files:
                #print(filename)
                #regexp for only the base csv files that has only numbers in them
                if(re.search("^csvfile-[0-9]*-[0-9]*.csv",filename)) is not None:
                    tmp_res[r].append(base_path+r+"/"+filename)

    if(len(tmp_res) == 0):
        print("Something happened during reading the csv files...Maybe the naming convention (csvfile-XXX-XXX.csv) is invalid?")
        exit(-1)
    # print(tmp_res)
    return _load_all_data(tmp_res)

def _load_all_data(list_of_files_with_explicit_path):
    df_count = 0
    df=None
    #print(list_of_files_with_explicit_path)
    for resolver_files_tuple in list_of_files_with_explicit_path.items():
        resolver=resolver_files_tuple[0]
        filenames=resolver_files_tuple[1]
        # print("filenames in _load_all_data:\n{}".format(filenames))
        if(resolver == ""):
            #this happens if the given list has accidentally a comma at the end
            #we need to handle this otherwise the program will crash just at the end :(
            print("There is an EMPTY resolver in the list...You might have put a ',' accidentally at the end of the list (?)")
            print("SKIPPING EMPTY resolver...")
            continue

        for i,f in enumerate(filenames):
            print("Processing {}: file #{} out of {} ({})".format(resolver,(i+1), len(filenames), f))
            #create a dataframe
            df_tmp = create_dataframe(f,resolver)

            print("Merging into dataframe...")
            if df_count == 0: #first time, we only have one df
                df=df_tmp
                df_count = 1
            else:
                #we already have a dataframe, so we just append
                df=pd.concat([df, df_tmp])
#            print("[DONE]")
    if df is None:
        print("Something happened during file processing and no dataframe is built")
        print("Exiting...")
        exit(-1)
    #reset indices
    df=df.reset_index(drop=True)
    # print(df)
    # train_df = df[['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_pkt_time_lag', 'seq_no_diff', 'Label']]
    train_df = df

    # test_df  =  df[['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_pkt_time_lag', 'seq_no_diff']]
    # test_df  =  df[['pkt_len']] #, 'prev_pkt_len']]
    # label    = df['Label']

    # RANDOM UNDERSAMPLING to create BALANCED DATASET
    #class count
    doh = df[df['Label']==1]
    web = df[df['Label']==0]
    print("#DoH packets: {}".format(len(doh)))
    print("#Web packets: {}".format(len(web)))

    # print('Before Random under-sampling:')
    # print(df.Label.value_counts())
    # count_class_doh, count_class_web = df.Label.value_counts()
    # #divide by class
    # df_class_doh = df[df['Label'] == 1]
    # df_class_web = df[df['Label'] == 0]
    # df_class_web_under = df_class_web.sample(count_class_doh)
    # df_test_under = pd.concat([df_class_web_under, df_class_doh], axis=0)
    # print('Random under-sampling:')
    # print(df_test_under.Label.value_counts())
    # exit(-1)

    return train_df

def train_and_test(dataframe, load_model=LOAD_MODEL):
    '''
    This is one of the most important function. It trains a model from scratch or loads an already built one.
    @params
    dataframe panda.dataframe - Can either be a dataframe or None. If None, it implicitly means OpenWorld scenario and model to be loaded
    load_model Boolean - True if model has to be loaded, False otherwise
    '''

    if load_model:
        #splitting dataframe for training and testing
        if (CLOSED_WORLD == True):
            #dataframe training and testing dataset is required later for ROC curves
            df_test = dataframe[['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_pkt_time_lag']]
            Label=dataframe['Label']
            x_train, x_test, y_train, y_test = train_test_split(df_test, Label, test_size=TEST_SIZE, random_state=109)

            print("Testing on {}".format(DATAFRAME_NAME))
        else:
            if(OPEN_WORLD is not None):
                print("Testing on {}".format(OPEN_WORLD))
            else:
                print("Testing on {}".format(OPEN_WORLD_RAW_DATA))

        print("Loading model from {}".format(MODEL_NAME_WITH_FULL_PATH))
        try:
            rfc=joblib.load(MODEL_NAME_WITH_FULL_PATH)
            # print("[DONE]")
        except FileNotFoundError:
            print("Model {} not found! Are you sure you set its name correctly? EXITING...".format(MODEL_NAME))
            exit(-1)
            # print("Let's retrain from scratch...")
            # #call recursively this same function by explicitly branching to the else branch
            # train_and_test(dataframe, False)


    else:
    #     print("Training and testing...")
    #     if(dataframe is None): #SHOULD NEVER HAPPEN, BUT LET'S CHECK
    #         print("Dataframe is None, however it is set to load one or build one from scratch")
    #         print("UNRESOLVED ERROR....EXITING")
    #         exit(-1)

        #splitting dataframe for training and testing
        df_test = dataframe[['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_pkt_time_lag']]
        Label=dataframe['Label']
        x_train, x_test, y_train, y_test = train_test_split(df_test, Label, test_size=TEST_SIZE, random_state=109)

        #training
        rfc = RandomForestClassifier(n_estimators = NUM_TREES, criterion="entropy", verbose=2, n_jobs=CPU_CORES)
        rfc.fit(x_train,y_train)

        if(MODEL_NAME_WITH_FULL_PATH is not None):
            print("Saving latest model as {}".format(MODEL_NAME_WITH_FULL_PATH))
            try:
                joblib.dump(rfc, MODEL_NAME_WITH_FULL_PATH,compress=('xz',3)) #compress with xz with compression level 3
            except OSError as e:
                print("There was an error during saving the model ({})...SKIPPING".format(MODEL_NAME))
                print(str(e))
                try:
                    os.remove(MODEL_NAME_WITH_FULL_PATH) #remove file is a portion of it has been saved
                    print("Half ready model has been deleted!")
                except:
                    print("Error during deleting half-ready model file {}! Probably it was not even created? Don't care, continue".format(MODEL_NAME))
            print("[DONE]")

    f = open(OUTPUT_DIR+"/"+"accuracy_metrics_"+META+".nfo", "a")
    f.write(str("Dataframe: {}\n".format(DATAFRAME_NAME_WITH_FULL_PATH)))
    f.write(str("Model (is saved): {}\n".format(MODEL_NAME)))
    f.write(str("Meta: {}\n".format(META)))



    if(CLOSED_WORLD): #closed world scenario
        print("+=======================+")
        print("| CLOSED WORLD SETTINGS |")
        print("+=======================+")

        rfc_pred=rfc.predict(x_test)

        print("Accuracy :",metrics.accuracy_score(y_test, rfc_pred ))
        print("Precision:",metrics.precision_score(y_test,rfc_pred ))
        print("Recall:",metrics.recall_score(y_test, rfc_pred))
        print("F1 Score:",metrics.f1_score(y_test,rfc_pred ))
        print("Confusion Matrix :\n" ,metrics.confusion_matrix(y_test,rfc_pred))

        # print("Metrics:")
        # for i in metrics:
        #     print(i)

        f.write("+=======================+\n")
        f.write("| CLOSED WORLD SETTINGS |\n")
        f.write("+=======================+\n")
        f.write(str("Accuracy: {}\n".format(metrics.accuracy_score(y_test, rfc_pred))))
        f.write(str("Precision:{}\n".format(metrics.precision_score(y_test,rfc_pred))))
        f.write(str("Recall:{}\n".format(metrics.recall_score(y_test, rfc_pred))))
        f.write(str("F1 Score:{}\n".format(metrics.f1_score(y_test,rfc_pred))))
        f.write(str("Confusion Matrix:{}\n".format(metrics.confusion_matrix(y_test,rfc_pred))))
        return {"model":rfc, "x_test":x_test, "y_test":y_test, "open_world_dataframe":None}
    else:
        open_world_dataframe_name = None
        if(OPEN_WORLD is not None and OPEN_WORLD_RAW_DATA is None):
            #IF open world data should be read from dataframe file
            #f.write(str("Tested on: {}\n".format(OPEN_WORLD)))
            print("Dataframe for open-world testing")
            #load the dataframe for the open-world setting
            try:
                #we need to define new variables with new names to avoid any collision with the ones above
                print("\tLoading from {}...".format(OPEN_WORLD))
                open_world_dataframe=pd.read_pickle(OPEN_WORLD)
                open_world_dataframe_name = OPEN_WORLD.split("/")[-1]
            except FileNotFoundError:
                print("Open-world dataframe {} not found! Exiting...".format(OPEN_WORLD))
                exit(-1)
        elif (OPEN_WORLD is None and OPEN_WORLD_RAW_DATA is not None):
            #open world data is defined by path to raw data, we have to create dataframe from scratch
            #OPEN_WORLD_RAW_DATA is path to a directory! we need the last substring after '/' - this will be the resolver name
            resolver = OPEN_WORLD_RAW_DATA.split("/")[-1] #we need a list because of the get_data_for_training(), so we surround it with []
            #then we remove the resolver from the full path will result in the path to the directory one layer above
            path = OPEN_WORLD_RAW_DATA.replace(resolver,"")
            open_world_dataframe_name = resolver
            #we have to convert  resolver to a list to use this function below
            resolver = [resolver]
            print("\tCreating from scratch...")
            open_world_dataframe=get_data_for_training(resolver, path)

        else: #both are not None
            pass #this is handled in argparse

        f.write(str("Tested on: {}\n".format(OPEN_WORLD_RAW_DATA)))


        print("Testing...")
        #Data frame loaded, no reason for the try-catch
        open_world_x_test = open_world_dataframe[['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_pkt_time_lag']]
        open_world_y_test = open_world_dataframe['Label']

        open_world_rfc_pred=rfc.predict(open_world_x_test)


        print("+=======================+")
        print("| OPEN WORLD SETTINGS |")
        print("+=======================+")
        print("Accuracy :",metrics.accuracy_score(open_world_y_test, open_world_rfc_pred ))
        print("Precision:",metrics.precision_score(open_world_y_test,open_world_rfc_pred ))
        print("Recall:",metrics.recall_score(open_world_y_test, open_world_rfc_pred))
        print("F1 Score:",metrics.f1_score(open_world_y_test,open_world_rfc_pred ))
        print("Confusion Matrix :\n" ,metrics.confusion_matrix(open_world_y_test,open_world_rfc_pred))

        f.write("+=======================+\n")
        f.write("| OPEN WORLD SETTINGS |\n")
        f.write("+=======================+\n")
        f.write(str("Accuracy: {}\n".format(metrics.accuracy_score(open_world_y_test, open_world_rfc_pred))))
        f.write(str("Precision:{}\n".format(metrics.precision_score(open_world_y_test,open_world_rfc_pred))))
        f.write(str("Recall:{}\n".format(metrics.recall_score(open_world_y_test, open_world_rfc_pred))))
        f.write(str("F1 Score:{}\n".format(metrics.f1_score(open_world_y_test,open_world_rfc_pred))))
        f.write(str("Confusion Matrix:{}\n".format(metrics.confusion_matrix(open_world_y_test,open_world_rfc_pred))))
        f.write(str("------------------------------ END -----------------------------\n\n\n\n"))
        f.close()

        return {"model":rfc, "x_test":open_world_x_test, "y_test":open_world_y_test, "open_world_dataframe":open_world_dataframe, "open_world_dataframe_name":open_world_dataframe_name}

def make_shap(model,x_test,y_test,basename):
    explainer = shap.TreeExplainer(model)
    select = range(5)
    pyplot.clf()
    features = x_test.iloc[select]
    # print(features)
    features_display = x_test.loc[features.index]
    # print(features_display)
    labels = y_test.iloc[select]
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

    # meta=MODEL_NAME
    # if OPEN_WORLD is not None: #to store the open world dataframe as well in the filename
    #     meta=meta + "_" + os.path.basename(OPEN_WORLD)

    pyplot.savefig(OUTPUT_DIR+"/"+basename+META+".shapley.pdf")
    pyplot.savefig(OUTPUT_DIR+"/"+basename+META+".shapley.png")

def generate_histogram(dataframe, bin_start, bin_stop, bin_step, column_name, filename_base, xlabel="Packet value", ylabel="Number of packets"):
    print("Generating histogram...")

    bin_values=np.arange(start=bin_start,stop=bin_stop,step=bin_step)
    doh = dataframe[dataframe['Label']==1]
    try:
        doh = doh[column_name]
    except:
        print("There was an error during creating the histogram for feature {}".format(column_name))
        print("Probably missing from the dataframe...SKIPPING")
        return
    # print(doh)
    web = dataframe[dataframe['Label']==0]
    web = web[column_name]

    pyplot.clf()
    fig=pyplot.figure()
    pyplot.hist(x=web, bins=bin_values, alpha=0.5, label="Web")
    pyplot.hist(x=doh, bins=bin_values, alpha=0.5, label="DoH")
    pyplot.legend(loc='upper right')

    # fig.subtitle('test title', fontsize=20)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.savefig(OUTPUT_DIR+"/"+filename_base+".pdf")
    pyplot.savefig(OUTPUT_DIR+"/"+filename_base+".png")

def generate_roc_auc_csv(df_test, Label, model, resolver=RESOLVERS_FOR_TRAINING[0], basename="ROC_AUC_"+META, max_fpr=None):

    lr_probs = model.predict_proba(df_test)
    if max_fpr is None:
        print("ROC score (partial AUC): {}".format(roc_auc_score(Label, lr_probs[:,1],max_fpr=max_fpr)))
    else:
        print("ROC score (for max FPR {}): {}".format(max_fpr, roc_auc_score(Label, lr_probs[:,1],max_fpr=max_fpr)))

    fpr, tpr, _ = roc_curve(Label, lr_probs[:,1])

    basename=OUTPUT_DIR + "/" +basename
    csv_file  = basename + ".csv"
    plot_file = basename + ".pdf"
    plot_file_png = basename + ".png"
    plot_label = resolver


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

def generate_pr_csv(df_test, Label, model, resolver=RESOLVERS_FOR_TRAINING[0], basename="PRC_"+META):
    """
    @params
    df_test Dataframe - testing data set (without label), i.e., x_test
    Label Dataframe - Labels, i.e., y_test
    model ML_MODEL - the trained model's instance
    resolver Str - resolver name for easier identification and labeling the plots
    basename Str - extra basename for even much easier identification for the output filenames
    """
    lr_probs = model.predict_proba(df_test)
    # print("lr_probs:{}".format(lr_probs))
    lr_probs = lr_probs[:, 1]
    # print("lr_probs[:, 1]:{}".format(lr_probs))
    lr_precision, lr_recall, threshold = precision_recall_curve(Label, lr_probs)

    basename=OUTPUT_DIR + "/" +basename
    csv_file  = basename + ".csv"
    plot_file = basename + ".pdf"
    plot_file_png = basename + ".png"
    plot_label = resolver

    # print("Threshold for PRC: {}".format(threshold))
    prc_df = pd.DataFrame({'precision' : lr_precision, 'recall':lr_recall})
    prc_df.to_csv(csv_file, index_label='index')

    pyplot.clf()
    fig=pyplot.figure()
    pyplot.rcParams["figure.figsize"]=5,5
    no_skill = len(Label[Label==1]) / len(Label)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label=plot_label)
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.savefig(plot_file)
    pyplot.savefig(plot_file_png)


###########################################
#               MAIN                      #
###########################################


#making training dataframe
print("Dataframe for training")
dataframe=make_dataframe(load_dataframe=LOAD_DATAFRAME)
if CREATE_DATAFRAME_ONLY:
    print("Dataframe created and saved!")
    print("There is nothing else to do for now...EXITING")
    exit(0)

if(WRITE_DATAFRAME_TO_CSV):
    print("Saving dataframe as csv file ({}). Note, Label=1 means DoH".format(str(OUTPUT_DIR+"/"+DATAFRAME_NAME+".csv")))
    df_to_write=dataframe[['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_pkt_time_lag', 'Label']]
    df_to_write.to_csv(OUTPUT_DIR + "/" + DATAFRAME_NAME+".csv",index_label='index')
    print("[DONE]")
# TRAIN AND TEST
results=train_and_test(dataframe, LOAD_MODEL)
model=results["model"]
x_test=results["x_test"]
y_test=results["y_test"]
if(CLOSED_WORLD == False):
    open_world_dataframe = results["open_world_dataframe"]
    open_world_dataframe_name = results["open_world_dataframe_name"]
else:
    print("Training - Testing ratio: {} - {}".format((1-TEST_SIZE), TEST_SIZE))

if(HISTOGRAM):
    if(dataframe is None): #this case, no dataframe was loaded for training, so no histogram can be made
        print("Model was loaded without the need of a dataframe! Histogram for the data used for the model cannot be made!")
    else:
        #histogram generations
        generate_histogram(dataframe, 50, 300, 1, "pkt_len", "histogram_pkt_len_"+DATAFRAME_NAME, xlabel="Packet Size [B]")
        generate_histogram(dataframe, 50, 300, 1, "prev_pkt_len", "histogram_prev_pkt_len_"+DATAFRAME_NAME, xlabel="Packet Size [B]")
        generate_histogram(dataframe, 0, 0.0005, 0.000001, "time_lag", "histogram_time_lag_"+DATAFRAME_NAME, "Time lag [s]")
        generate_histogram(dataframe, 0, 0.0005, 0.000001, "prev_pkt_time_lag", "histogram_prev_time_lag_"+DATAFRAME_NAME, "Time lag [s]")

    # generate_histogram(dataframe, 0, 200, 1, "seq_no_diff", "histogram_seq_no_diff_"+DATAFRAME_NAME, xlabel="Sequence no. Difference")
    if(CLOSED_WORLD == False): #there is any open-world dataframe after testing, we make histograms for that as well
        generate_histogram(open_world_dataframe, 50, 300, 1, "pkt_len", "histogram_pkt_len_"+DATAFRAME_NAME+"_ow_"+open_world_dataframe_name, xlabel="Packet Size [B]")
        generate_histogram(open_world_dataframe, 50, 300, 1, "prev_pkt_len", "histogram_prev_pkt_len_"+DATAFRAME_NAME+"_ow_"+open_world_dataframe_name, xlabel="Packet Size [B]")
        generate_histogram(open_world_dataframe, 0, 0.0005, 0.000001, "time_lag", "histogram_time_lag_"+DATAFRAME_NAME+"_ow_"+open_world_dataframe_name, "Time lag [s]")
        generate_histogram(open_world_dataframe, 0, 0.0005, 0.000001, "prev_pkt_time_lag", "histogram_prev_time_lag_"+DATAFRAME_NAME+"_ow_"+open_world_dataframe_name, "Time lag [s]")

if(PRC):
    # CREATE PRC CURVES
    print("Generating PRC curve")
    if(dataframe is not None): #there was dataframe created/loaded
        basename="PRC_"+DATAFRAME_NAME+META
    else: #there was NO dataframe created/loaded so use model name instead
        basename="PRC_"+MODEL_NAME+META
    if(CLOSED_WORLD == False): #to store the open world dataframe as well in the filename
        resolver=os.path.basename(OPEN_WORLD)
    generate_pr_csv(df_test=x_test,Label=y_test, model=model, resolver="",basename=basename)

if(ROC_AUC):
    # CREATE PRC CURVES
    print("Generating ROC curve")
    if(dataframe is not None): #there was dataframe created/loaded
        basename="ROC_AUC_"+DATAFRAME_NAME+META
    else: #there was NO dataframe created/loaded so use model name instead
        basename="ROC_AUC_"+MODEL_NAME+META
    generate_roc_auc_csv(df_test=x_test,Label=y_test, model=model, resolver="",basename=basename, max_fpr=MAX_FPR)

if(SHAPLEY):
    # CREATE SHAP VALUES
    print("Creating SHAP values")
    if(dataframe is not None): #there was dataframe created/loaded
        basename="SHAP_"+DATAFRAME_NAME
    else: #there was NO dataframe created/loaded so use model name instead
        basename="SHAP_"+MODEL_NAME
    make_shap(model, x_test, y_test,basename)
