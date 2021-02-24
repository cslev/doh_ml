#!/usr/bin/python3

from logger import Logger #for custom logging

import random  #for getting a random snippet from the dataframe at the end

import argparse #for argument parsing
import argcomplete #for BASH autocompletion

import pandas as pd #for pandas
import numpy as np #for padding

from os import path,walk #for checking paths, dirs, and looping through the files within a dir
import re # regexp for checking csv files convention
import sys # for writing in the same line via sys.stdout

from sklearn.utils import resample #for balancing the data set

parser = argparse.ArgumentParser(description="Create Panda dataframes for doh_ml from " +
                                ".csv files generated via doh_docker container", 
                                formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('-i',
                    '--input',
                    action="store",
                    metavar='N',
                    type=str,
                    required=True,
                    nargs='+',
                    dest="traffic_traces",
                    help="Specify here the path to the .csv files containing the traffic" +
                    "traces. \n" +
                    "Use /path/to/dir to load all .csv files within the given dir.\n" +
                    "Or, use /full/path/to/csvfile-1-200.csv to load one file only.\n"+
                    "Or, use multiple paths to select " +
                    "(and combine) multiple .csv files or whole directories. \n" +
                    "Example: -i /path/to/dir /full/path/to/csvfile-1-200.csv /path/to/dir2")

#For balancing the dataset...not recommended though, results are quite strange (1.0),
#and k-fold cross-validation was okay for all use cases so far
parser.add_argument('-B',
                    '--balance-dataset',
                    action="store_true",
                    default=False,
                    dest="balance_dataset",
                    help="Specify whether to balance the dataset (Default: False")


parser.add_argument('-p',
                    '--pad-packet-len',
                    action="store",
                    default=None,
                    type=int,
                    dest="pad_pkt_len",
                    help="Specify whether to pad each DoH packet's pkt_len and how (Default: no padding)\n" +
                    "1: Pad according to RFC 8467, i.e., to the closest multiple of 128 bytes\n" +
                    "2: Pad with a random number between (1,MTU-actual packet size)\n" +
                    "3: Pad to a random number from the distribution of the Web packets\n" +
                    "4: Pad to a random preceding Web packet's size\n" +
                    "5: Pad a sequence of DoH packets to a random sequence of preceeding Web packets' sizes\n")
parser.add_argument('-t',
                    '--pad-time-lag',
                    action="store",
                    default=None,
                    type=int,
                    dest="pad_time_lag",
                    help="Specify whether to pad each DoH packet's time_lag and how (Default: no padding)\n" +
                    "3: Pad to a random number from the distribution of the Web packets\n" +
                    "4: Pad to a random preceding Web packet's size\n" +
                    "5: Pad a sequence of DoH packets to a random sequence of preceeding Web packets' sizes\n")

parser.add_argument('-b',
                    '--bidir',
                    action="store_true",
                    dest='bidir',
                    default=False,
                    help="Specify if dataframe should be bidirectional. \n" +
                    "Default: False (only requests will be present")

parser.add_argument('-o',
                    '--output',
                    action="store",
                    default="./dataframes/df.pkl",
                    type=str,
                    dest="dataframe_path",
                    help="Specify the full path for the " +
                    "dataframe to be saved (Default: ./dataframes/df.pkl)")



#for BASH autocomplete  
argcomplete.autocomplete(parser)

args=parser.parse_args()


#TRAFFIC_TRACE_PATHS=args.traffic_traces.split(",")
TRAFFIC_TRACE_PATHS=args.traffic_traces
DATAFRAME=args.dataframe_path
SINGLE_TRACE=None
BALANCE_DATASET = args.balance_dataset

#PADDING related
PAD_PKT_LEN=args.pad_pkt_len
PAD_TIME_LAG=args.pad_time_lag
# further variables required for the different padding techniques
iterator = 0 # for padding techniques 4 and 5
prev_http_pkt_len = 0 # for padding techniques 4 and 5
prev_http_time_lag = 0.0 # for padding techniques 4 and 5
is_http_new = True # for padding techniques  5
prev_http_packets = [] # for padding techniques 4 and 5
# for padding techniques 4 and 5
WEBPACKET_BUFFER_SIZE_PKTLEN = 30 #empirically the best among 5,10,20,30
WEBPACKET_BUFFER_SIZE_TIMELAG = 50 #empirically the best among 5,10,20,30
RFC_PADDING=dict()
for i in range(1,12):
    RFC_PADDING[i]=i*128
RFC_PADDING[12] = 1500
RANDOM_VALUE_NORMAL_DISTRIBUTION_COUNTER = 5

#Packets are still bidirectional in the dataframe?
BIDIR=args.bidir


SELF="create_dataframe.py"
logger = Logger(SELF)

traffic_traces=list()


#used for progress tracking in lambda functions
lambda_counter = 0
number_of_elements = 0

#quick checks for the paths
logger.log_simple("Checking paths...")
for p in TRAFFIC_TRACE_PATHS:
  logger.log(p)
  if (not path.exists(p)):
    logger.log(p,logger.FAIL)
    logger.log("Exiting...")
    exit(-1)

  if path.isfile(p):
    logger.log(str("File {}...".format(p)),logger.FOUND)
    #Second parameter helps us to know that this path is for one file only
    traffic_traces.append((p,True))


  if path.isdir(p):
    # print("\tDirectory of .csv files found".format(p))  
    logger.log(str("Directory {}...".format(p)),logger.FOUND)
    #Second parameter helps us to know that this path is for a whole directory
    traffic_traces.append((p,False))


### PADDING RELATED FUNCTIONS
def get_random_choice_from_list(list_of_values, value_to_be_greater_than):
  '''
  Get a random element from a list containing previous packets' values. However,
  the randomness is not full randomness. A value is only returned if it is bigger
  than the current one. Eventually, this is padding, so smaller cannot be returned :)
  List list_of_values  - previous elements
  Int/Float value_to_be_greater_than - self-explanatory
  '''
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
  '''
  Get the next element from a list that is greater than a given value passed as
  argument.
  List list_of_values  - previous elements
  Int/Float value_to_be_greater_than - self-explanatory
  Iterator iterator - can be used to indicate where to start the process. The 
  iterator itself is also return to the caller function
  '''
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

def get_random_from_normal_distribution(mean, stdev):
  '''
  Get a random value from a normal distribution defined by its mean and stdev.
  The reason of having this function as a separate one is to be able to call
  it recursively if needed, i.e., when the random value is smaller than the one
  we want to use.
  Float mean - the mean value
  Float stdev - the standard deviation
  returns float - a random value
  '''
  return np.random.normal(mean,stdev)

def _lambda_pad_pkt_len(pkt_len, Label, mean, stdev):
  '''
  Padding pkt_len values according to well-defined schemes. See the help output
  for more details.
  The function is called for each row of the dataframe via a lambda function.
  '''
  global prev_http_pkt_len
  global prev_http_packets
  global is_http_new
  global iterator

  #for keeping track of the process
  global lambda_counter
  

  pkt_len=int(pkt_len)
  #DoH PACKET
  if(int(Label) == 1):
    # print(PAD_PKT_LEN, type(PAD_PKT_LEN))
    if(PAD_PKT_LEN == 1):
    # RFC 8467
      # if, by any chance the packet would be higher than 1500 bytes, we don't do anything
      if(pkt_len < 1500): 
        index = int(pkt_len/128) + 1 #get the index for the RFC_PADDING list
        #pad each packet to the closest multiple of 128B
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
        #stdev is usually bigger than mean, we have to use abs otherwise pkt_len
        #becomes a negative number
        new_pkt_len = abs(int(get_random_from_normal_distribution(mean,stdev))) 
        if(new_pkt_len < pkt_len):
          new_pkt_len = pkt_len + new_pkt_len
          if(new_pkt_len > 1500):
            new_pkt_len = pkt_len
      else:
        new_pkt_len = pkt_len
        


    # pad to a random picked preceding Web packet's size
    elif(PAD_PKT_LEN == 4):
      #we need to have more than 2 previous Web packets in our list
      if len(prev_http_packets) > 2: 
        #get a random packet from the previous Web packets
        new_pkt_len = get_random_choice_from_list(prev_http_packets, pkt_len)
        #if there is no greater element than current pkt_len
        if new_pkt_len == None: 
          #simply apply technique 3 (PAD_PKT_LEN = 3)
          new_pkt_len = abs(int(get_random_from_normal_distribution(mean,stdev))) 
          if(new_pkt_len < pkt_len):
            new_pkt_len = pkt_len + new_pkt_len
            if(new_pkt_len > 1500):
              new_pkt_len = 1500
      #if the number of preceeding web packets is less than 3, apply technique 3
      else:        
        #simply apply technique 3 (PAD_PKT_LEN = 3)
        new_pkt_len = abs(int(get_random_from_normal_distribution(mean,stdev))) 
        if(new_pkt_len < pkt_len):
          new_pkt_len = pkt_len + new_pkt_len
          if(new_pkt_len > 1500):
            new_pkt_len = 1500

    # pad to a randomly selected sequence of the preceding Web packets' sizes
    elif(PAD_PKT_LEN == 5):
      # iterator = 0
      #we need to have more than 2 previous Web packets in our list
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
            #simply apply technique 3 (PAD_PKT_LEN = 3)
            new_pkt_len = abs(int(get_random_from_normal_distribution(mean,stdev))) 
            if(new_pkt_len < pkt_len):
              new_pkt_len = pkt_len + new_pkt_len
              if(new_pkt_len > 1500):
                new_pkt_len = 1500

      #this applies only in the beginning, when there is not enough Web packets from the past
      else:
        #simply apply technique 3 (PAD_PKT_LEN = 3)
        new_pkt_len = abs(int(get_random_from_normal_distribution(mean,stdev))) 
        if(new_pkt_len < pkt_len):
          new_pkt_len = pkt_len + new_pkt_len
          if(new_pkt_len > 1500):
            new_pkt_len = 1500

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
    
    # return the original value for Web packets
    return pkt_len


def _lambda_pad_time_lag(time_lag, Label, mean, stdev):
  '''
  Padding time_lag values according to well-defined schemes. See the help output
  for more details.
  The function is called for each row of the dataframe via a lambda function.
  '''
  global prev_http_pkt_len
  global prev_http_time_lag
  global prev_http_packets
  global is_http_new
  global iterator
  if(int(Label) == 1):
    # Pad to a random size from the normal distribution of the web traffic
    if(PAD_TIME_LAG == 3):
      #get a random padding amount
      #stdev is usually bigger than mean, we have to use abs otherwise pkt_len
      #becomes a negative number
      new_time_lag = abs(float(get_random_from_normal_distribution(mean,stdev)))
      #if time_lag is not bigger than the new time_lag, then use the new one
      if(new_time_lag < time_lag):
        new_time_lag = new_time_lag + time_lag 
      

    # pad to a random picked preceding Web packet's size
    elif(PAD_TIME_LAG == 4):
      #we need to have more than 2 previous Web packets in our list
      if len(prev_http_packets) > 2:
        new_time_lag = get_random_choice_from_list(prev_http_packets, time_lag)
        if new_time_lag == None: #if there is no greater element than current time_lag
          #simply apply technique 3 (PAD_TIME_LAG = 3)
          new_time_lag = abs(float(get_random_from_normal_distribution(mean,stdev)))
          #if time_lag is not bigger than the new time_lag, then use the new one
          if(new_time_lag < time_lag):
            new_time_lag = new_time_lag + time_lag 
      else:
        #simply apply technique 3 (PAD_TIME_LAG = 3)
        new_time_lag = abs(float(get_random_from_normal_distribution(mean,stdev)))
        #if time_lag is not bigger than the new time_lag, then use the new one
        if(new_time_lag < time_lag):
          new_time_lag = new_time_lag + time_lag 

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
            new_time_lag = abs(float(get_random_from_normal_distribution(mean,stdev)))
            if(new_time_lag < time_lag):
              new_time_lag = new_time_lag + time_lag 

      #this applies only in the beginning, when there is not enough Web packets from the past
      else:
        new_time_lag = abs(float(get_random_from_normal_distribution(mean,stdev)))
        #if time_lag is not bigger than the new time_lag, then use the new one
        if(new_time_lag < time_lag):
          new_time_lag = new_time_lag + time_lag 

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
  '''
  This is the main function for padding! It calls the relevant sub-functions
  according to the padding schemes selected.
  '''
  global prev_http_pkt_len
  global prev_http_time_lag
  global is_http_new
  global prev_http_packets
  web = df_temp[df_temp["Label"] == 0]
  doh = df_temp[df_temp["Label"] == 1]
  if(pkt_len):
    prev_http_pkt_len=0
    is_http_new=True
    prev_http_packets=[]

    pkt_len_mean  = web.pkt_len.mean()
    pkt_len_stdev = web.pkt_len.std()
    logger.log_simple(" => Statistics of Web packets:")
    logger.log_simple(" ---> WEB pkt_len_mean:  {}".format(pkt_len_mean))
    logger.log_simple(" ---> WEB pkt_len_stdev: {}".format(pkt_len_stdev))
    logger.log_simple(" ---> DoH pkt_len_mean:  {}".format(doh.pkt_len.mean()))
    logger.log_simple(" ---> DoH pkt_len_stdev: {}".format(doh.pkt_len.std()))
    # print(df_temp)
    logger.log("Padding pkt_len values with technique {}...".format(pkt_len))
    df_temp['pkt_len']= df_temp.apply(lambda x : _lambda_pad_pkt_len(x.pkt_len, 
                                                              x.Label, 
                                                              pkt_len_mean, 
                                                              pkt_len_stdev), 
                                                              axis=1)
    logger.log("Padding pkt_len values with technique {}...".format(pkt_len), logger.OK)
    #print out some stats after padding DoH packets
    doh = df_temp[df_temp["Label"] == 1]
    logger.log_simple(" => Statistics of DoH packets after padding:")
    logger.log_simple(" ---> DoH pkt_len_mean:  {}".format(doh.pkt_len.mean()))
    logger.log_simple(" ---> DoH pkt_len_stdev: {}".format(doh.pkt_len.std()))
  else:
    logger.log("Padding pkt_len values...", logger.SKIP)

  # print(df_temp)
  if(time_lag):
    prev_http_time_lag=0
    is_http_new=True
    prev_http_packets=[]

    time_lag_mean = web.time_lag.mean()
    time_lag_stdev = web.time_lag.std()

    logger.log_simple(" => Statistics of Web packets:")
    logger.log_simple(" ---> WEB time_lag_mean:  {}".format(time_lag_mean))
    logger.log_simple(" ---> WEB time_lag_stdev: {}".format(time_lag_stdev))
    logger.log_simple(" ---> DoH time_lag_mean:  {}".format(doh.time_lag.mean()))
    logger.log_simple(" ---> DoH time_lag_stdev: {}".format(doh.time_lag.std()))

    logger.log("Padding time_lag values with technique {}...".format(time_lag))
    df_temp['time_lag'] = df_temp.apply(lambda x : _lambda_pad_time_lag(x.time_lag, 
                                                                x.Label, 
                                                                time_lag_mean, 
                                                                time_lag_stdev), 
                                                                axis=1)
    logger.log("Padding time_lag values with technique {}...".format(time_lag), logger.OK)

    #print out some stats after padding DoH packets
    doh = df_temp[df_temp["Label"] == 1]
    logger.log_simple(" => Statistics of DoH packets after padding:")
    logger.log_simple(" ---> DoH time_lag_mean:  {}".format(doh.time_lag.mean()))
    logger.log_simple(" ---> DoH time_lag_stdev: {}".format(doh.time_lag.std()))
  else:
    logger.log("Padding time_lag values...", logger.SKIP)

  return df_temp


### END PADDING RELATED FUNCTIONS


def load_csvfile(filename):
  '''
  This function simply takes a file and makes a dataframe from it
  '''
  ### the columns in the new .csv files
  rename_table = {
    "frame.numer": "No.",
    "_ws.col.Time": "Time",
    "ip.src":"src_ip",
    "ip.dst":"dst_ip",
    "tcp.srcport" : "src_port",
    "tcp.dstport" : "dst_port",
    "_ws.col.Protocol" : "Protocol",
    "frame_len" : "pkt_len",
    "_ws.col.Info": "Info"
  }

  #read and process .csv file
  df = pd.read_csv(filename, header=0, names=list(rename_table.keys()))

  #rename columns according to the rename_table
  df = df.rename(columns=rename_table)
  return df

def get_flow_id(src_ip,dst_ip,src_port,dst_port):
  '''
  This function is to generate a flow ID according to a common pattern
  '''
  return str("{}-{}:{}-{}".format(src_ip,dst_ip,src_port,dst_port))

def _lambda_add_label(protocol, info):
  '''
  This private function is a lambda function for add_label() function.
  This will replace the label of all Web packets having POST queries
  specific to DoH communication is relabeled to DoH
  '''
  #many doh resolvers have /dns-query as a last argument of the POST query
  #CleanBrowsing has /doh/family-filter/
  #powerDNS has / only
  HTTP_POST_QUERIES_TO_RELABEL = ["/dns-query", "/doh/family-filter/", "/"]
  if((protocol == "HTTP2") and (info.split(" ")[-1] in HTTP_POST_QUERIES_TO_RELABEL)): #Web that belongs to DoH
    return 1
  else: #there can only be HTTP2 and DoH, no other protocol since dataframe already filtered at this point
    if(protocol == "HTTP2"): #pure Web
      return 0
    else: # pure DoH
      return 1

def add_label(df) :
  '''
  This function will add labels with 0 (for Web) and 1 (for DoH). Also, all Web packets having POST queries
  specific to DoH communication is relabeled to DoH
  '''
  
  #will append dataframe with 'Label' column to 0 if value is HTTP2, else we either set it to 1 (if DoH) or -1 (otherwise)
  #let's change the remaining Web packet's label to DoH if they practically belong to a DoH communication
  new_df=df
  new_df.loc[:, "Label"]=new_df[["Protocol", "Info"]].apply(lambda x: _lambda_add_label(*x), axis=1)
  return new_df

#### 
# def add_flow_ids(df):
#   '''
#   Add flow IDs to the packets. This helps later to process data in a flow-based
#   level.
#   '''
#   new_df = df
  
#   #we extend the dataframe with flow IDs
#   flow_id_counter = 0 #flowIDs will be integer numbers
#   flow_id = flow_id_counter #first flow id is 0
#   flow_ids = dict() #to keep track of what flows did we see

#   #store the flow IDs in a list, this will be added to the dataframe as a new 
#   #column
#   flow_id_list = list() 
#   ### for progress tracking with logger.calculateRemainingPercentage()
#   current_id = 0
#   n=len(new_df)
#   #obtain relevant info
#   df_temp = new_df[["src_ip", "dst_ip", "src_port", "dst_port"]]

#   ## Getting the relevant cells for every row in the dataframe
#   for row in df_temp.itertuples(index=False):
#     src_ip=row.src_ip
#     dst_ip=row.dst_ip
#     src_port=row.src_port
#     dst_port=row.dst_port

#     current_id += 1
#     logger.calculateRemainingPercentage(current_id,n, "Identifying flows...")
#     ### Get/generate flowID
#     flow=get_flow_id(src_ip,dst_ip,src_port,dst_port)

#     if flow not in flow_ids: #unseen flows
#       flow_ids[flow]=flow_id_counter #set new flow ID
#       flow_id = flow_id_counter #actual flow ID
#       flow_id_counter+=1 #for the next flow ID

#     else: #previouslyseen flow
#       flow_id = flow_ids[flow] #get corresponding flow ID
    
#     #add flow ID to the flow_id_list
#     flow_id_list.append(flow_id)
    
  
#   new_df["flowID"]=flow_id_list
#   logger.log("Identifying flows...", logger.OK)

#   return new_df
####

def relabel_outlier_flows(df):
  '''
  This function takes the dataframe and updates every packets' label to DoH 
  within a flow, if there was at least one DoH packet for the same flow
  Note, we could not apply pure IP based relabeling as some DoH resolver shares 
  its IP with the corresponding Web service. So, we have to go flow-by-flow and
  analyze the packets 
  '''
  new_df = df
  ### To keep track of the flows and their last Time value
  flows = dict() #flowID - DoH?)
 
  #we extend the dataframe with flow IDs
  flow_id_counter = 0 #flowIDs will be integer numbers
  flow_id = flow_id_counter #first flow id is 0
  flow_ids = dict() #to keep track of what flows did we see

  #store the flow IDs in a list, this will be added to the dataframe as a new 
  #column
  flow_id_list = list() 
  

  ### for progress tracking with logger.calculateRemainingPercentage()
  current_id = 0
  n=len(new_df)

  df_temp = new_df[["src_ip", "dst_ip", "src_port", "dst_port", "Label"]]
  
  ## Getting the relevant cells for every row in the dataframe
  for row in df_temp.itertuples(index=False):
    src_ip=row.src_ip
    dst_ip=row.dst_ip
    src_port=row.src_port
    dst_port=row.dst_port
    label=row.Label
    
    current_id += 1
    logger.calculateRemainingPercentage(current_id,n, "Adding flowIDs and looking for outliers...")
    ### Get/generate flowID
    flow=get_flow_id(src_ip,dst_ip,src_port,dst_port)
    # print(flow)

    
    #previously unseen flow
    if flow not in flows: 
      flows[flow] = label #add the flow and initalize it with its first packet's label

    ## the flow has been already seen
    else: 
      if label == 1: #we change the label to 1 the packet analyzed now is DoH
        flows[flow] = label
      #we do not use an else condition here to set the label to 0 if it is Web
      #we only need one DoH packet to set the label to DoH (1) for good
      #then, this dictionary can be used below, to update each packet's label 
      #according to the whole flow's label it belongs to

    #Update flow ids
    if flow not in flow_ids:
      flow_ids[flow]=flow_id_counter
      flow_id = flow_id_counter
      flow_id_counter+=1

    else:
      flow_id = flow_ids[flow]
    #add flow ID to the flow_id_list
    flow_id_list.append(flow_id)
    
  
  new_df["flowID"]=flow_id_list
    
  logger.log("Adding flowIDs and looking for outliers...",logger.OK)
  # new_df.loc[:, "Label"]=new_df.loc[["src_ip", "dst_ip", "src_port", "dst_port"]].apply(lambda x: _lambda_update_label(*x), axis=1)   

  labels=list()
  current_id = 0 #reset for proper progress tracking
  for row in df_temp.itertuples(index=False):
    src_ip=row.src_ip
    dst_ip=row.dst_ip
    src_port=row.src_port
    dst_port=row.dst_port
    # label=row.Label
    
    current_id += 1
    logger.calculateRemainingPercentage(current_id, n, "Relabeling outlier flows...")

    flow = get_flow_id(src_ip,dst_ip,src_port,dst_port)

    if flow not in flows: #serious problem, it should never happen as just updated above
      logger.log_simple("Something went really bad during storing the flows...")
      logger.log_simple("This flow is missing: {}".format(flow))
      logger.log_simple("Exiting...")
      exit(-1)
    
    labels.append(flows[flow])
  
  #update Label column
  new_df["Label"]=labels

  logger.log("Relabeling outlier flows...", logger.OK)

  return new_df

def add_packet_direction(df):
  '''
  This function will extend the dataset with a 'direction' column according to the source IPs
  @TODO: If we had the destination port in the dataset, we could filter on it instead of this hacky way
  '''
  new_df = df
  #add new column 'direction' to every row, according to the Source column adjusted by the lambda function
  # new_df.loc[:, "direction"]=new_df["Source"].apply(lambda x: _lambda_add_packet_direction(x))
  
  #add new column 'direction' to every row, according to the src_port column adjusted by the lambda function
  new_df.loc[:, "direction"]=new_df["dst_port"].apply(lambda x: "request" if int(x)==443 else "response")
  return new_df

def adjust_dataframe(new_df, old_df, new_df_path):
  '''
  This function is for readjusting the dataframe of new_df according to old_df.
  Originally, every dataframe (i.e., every .csv file) starts from scratch 
  meaning that IDs and time starts from 0. Other features are fine.
  Therefore, we have to readjust the new frame according to the last row of the 
  older one
  Dataframe new_df - the dataframe to adjust
  Dataframe old_df - previous dataframe to adjust according to
  Str new_df_path - path to the actual .csv file for meaningful loggin purposes
  '''
  logger.log( "--> Adjusting relative time and numbering...")
  last_row  = old_df.tail(1)
  last_no   = int(last_row["No."])
  # print(last_no)
  last_time = float(last_row["Time"])
  new_df.loc[:, "Time"] = new_df["Time"].apply(lambda x: float(x) + last_time)
  new_df.loc[:, "No."]  = new_df["No."].apply(lambda  x: int(x) + last_no)
  logger.log( "--> Adjusting relative time and numbering...",logger.OK)
  sys.stdout.flush()
  return new_df 

def add_time_lag(df):
  '''
  Add 'time_lag' column to the dataframe according to the relative time difference
  to the previous packet of the same flow. Due to the same flow requirement, 
  this process can last long.
  '''
  new_df = df
  ### To keep track of the flows and their last Time value
  flows = dict() #flowID - last Time: value,  last pkt_len: value)
 
  ### for progress tracking with logger.calculateRemainingPercentage()
  current_id = 0
  n=len(new_df)

  df_temp = new_df[["src_ip", "dst_ip", "src_port", "dst_port", "Time"]]
  # logger.log_simple("Adding time lags and prev_pkt_lens...")

  time_lags = list()
  
  # #also add flowIDs to the dataframe
  # flow_id_counter = 0 #flowIDs will be integer numbers
  # flow_id = flow_id_counter #first flow id is 0
  # flow_ids = dict() #to keep track of what flows did we see
  # flow_id_list = list() 

  ## Getting the relevant cells for every row in the dataframe
  for row in df_temp.itertuples(index=False):
    src_ip=row.src_ip
    dst_ip=row.dst_ip
    src_port=row.src_port
    dst_port=row.dst_port
    time=row.Time
 
    current_id += 1
    logger.calculateRemainingPercentage(current_id,n, "Adding time_lag")
    ### Get/generate flowID
    flow=get_flow_id(src_ip,dst_ip,src_port,dst_port)
    # print(flow)

    if flow not in flows: #flow never seen
      flows[flow]=dict() #initialize a new sub_dict()
      flows[flow]["last_time"]=time #store actual row's Time value
      time_lag = 0.0

      #keeping track of flowIDs
      # flow_ids[flow]=flow_id_counter #set new flow ID
      # flow_id = flow_id_counter #actual flow ID
      # flow_id_counter+=1 #for the next flow ID

    else: #flow seen before
      ### data is stored in the dict as a tuple (last time, last pktlen)
      last_update_time = flows[flow]["last_time"] #get the last Time value
      time_lag = time - last_update_time #calculate time_lag
      flows[flow]["last_time"] = time #update last Time value

      #keeping track of flowIDs
      # flow_id = flow_ids[flow] #get corresponding flow ID    

    time_lags.append(time_lag)

    #add flow ID to the flow_id_list
    # flow_id_list.append(flow_id)

  #extend dataframe with the two new columns
  new_df["time_lag"]=time_lags
  # new_df["flowID"]=flow_id_list

  logger.log("Adding time_lag...",logger.OK)
  return new_df

def add_prev_pkt_len(df):
  '''
  Add 'prev_pkt_len' column to the dataframe according to the relative time difference
  to the previous packet of the same flow. Due to the same flow requirement, 
  this process can last long.
  '''
  new_df = df
  ### To keep track of the flows and their last Time value
  flows = dict() #flowID - last Time: value,  last pkt_len: value)
 
  ### for progress tracking with logger.calculateRemainingPercentage()
  current_id = 0
  n=len(new_df)

  df_temp = new_df[["src_ip", "dst_ip", "src_port", "dst_port", "pkt_len"]]
  # logger.log_simple("Adding time lags and prev_pkt_lens...")

  prev_pkt_lens = list()
  ## Getting the relevant cells for every row in the dataframe
  for row in df_temp.itertuples(index=False):
    src_ip=row.src_ip
    dst_ip=row.dst_ip
    src_port=row.src_port
    dst_port=row.dst_port
    pkt_len=row.pkt_len
 
    current_id += 1
    logger.calculateRemainingPercentage(current_id,n, "Adding prev_pkt_len")
    ### Get/generate flowID
    flow=get_flow_id(src_ip,dst_ip,src_port,dst_port)
    # print(flow)

    if flow not in flows: #flow never seen
      flows[flow]=dict() #initialize a new sub_dict()
      flows[flow]["last_pkt_len"] = pkt_len #store actual row's pkt_len value
      prev_pkt_len = 0

    else: #flow seen before
      ### data is stored in the dict as a tuple (last time, last pktlen)
      last_pkt_len = flows[flow]["last_pkt_len"] #get the last pkt_len

      prev_pkt_len = last_pkt_len

      flows[flow]["last_pkt_len"] = pkt_len #update last Time value
    

    prev_pkt_lens.append(prev_pkt_len)

  #extend dataframe with the two new columns
  new_df["prev_pkt_len"]=prev_pkt_lens

  logger.log("Adding prev_pkt_len...", logger.OK)
  return new_df

def add_prev_time_lag(df):
  '''
  Add 'prev_pkt_len' column to the dataframe according to the relative time difference
  to the previous packet of the same flow. Due to the same flow requirement, 
  this process can last long.
  '''
  new_df = df
  n=len(new_df)

  df_temp = new_df[["src_ip", "dst_ip", "src_port", "dst_port", "time_lag"]]
  current_id = 0 #reset for progress tracking
  flows = dict()
  prev_time_lags = list()
  for row in df_temp.itertuples(index=False):
    src_ip=row.src_ip
    dst_ip=row.dst_ip
    src_port=row.src_port
    dst_port=row.dst_port
    time_lag=row.time_lag

    current_id += 1

    logger.calculateRemainingPercentage(current_id,n, "Adding prev_time_lag")
    ### Get/generate flowID
    flow=get_flow_id(src_ip,dst_ip,src_port,dst_port)

    if flow not in flows:
      flows[flow] = time_lag #supposed to be 0.0
      last_time_lag= time_lag #supposed to be 0.0
    else:
      last_time_lag=flows[flow]
      flows[flow] = time_lag
    
    prev_time_lags.append(last_time_lag)

  #append to dataframe
  new_df["prev_time_lag"]=prev_time_lags

  logger.log("Adding prev_time_lag...",logger.OK)
  return new_df

def add_relative_features(df):
  '''
  This function will add a 'time_lag' column and 'prev_pkt_len' according to the 
  relative time difference and packet size difference to the previous packet of 
  the same flow. Due to the same flow requirement, this process can last longer 
  than others.
  We are going to iterate through the dataframe. For every row, we create a 
  flow ID and store the Time/pkt_len along. When a consecutive row belongs to a 
  flow we have already seen before, we obtain the last Time and pkt_len value of 
  that flow, calculate the difference, store the difference and, finally, update 
  the last seen flow ID's time/pkt_len with the latest corresponding flow's Time
  '''
  new_df = df
  ### To keep track of the flows and their last Time value
  flows = dict() #flowID - last Time: value,  last pkt_len: value)
 
  ### for progress tracking with logger.calculateRemainingPercentage()
  current_id = 0
  n=len(new_df)

  df_temp = new_df[["src_ip", "dst_ip", "src_port", "dst_port", "Protocol", "Time", "pkt_len"]]
  # logger.log_simple("Adding time lags and prev_pkt_lens...")

  # print(df_temp)
  time_lags = list()
  prev_pkt_lens = list()
  ## Getting the relevant cells for every row in the dataframe
  for row in df_temp.itertuples(index=False):
    src_ip=row.src_ip
    dst_ip=row.dst_ip
    src_port=row.src_port
    dst_port=row.dst_port
    time=row.Time
    pkt_len=row.pkt_len
 
    current_id += 1

    logger.calculateRemainingPercentage(current_id,n, "Adding time lags and prev_pkt_lens")
    ### Get/generate flowID
    flow=get_flow_id(src_ip,dst_ip,src_port,dst_port)
    # print(flow)

    if flow not in flows: #flow never seen
      flows[flow]=dict() #initialize a new sub_dict()
      flows[flow]["last_time"]=time #store actual row's Time value
      flows[flow]["last_pkt_len"] = pkt_len #store actual row's pkt_len value
      time_lag = 0.0
      prev_pkt_len = 0
    else: #flow seen before
      ### data is stored in the dict as a tuple (last time, last pktlen)
      last_update_time = flows[flow]["last_time"] #get the last Time value
      last_pkt_len = flows[flow]["last_pkt_len"] #get the last pkt_len

      time_lag = time - last_update_time #calculate time_lag
      prev_pkt_len = last_pkt_len

      flows[flow]["last_time"] = time #update last Time value
      flows[flow]["last_pkt_len"] = pkt_len #update last Time value
    

    time_lags.append(time_lag)
    prev_pkt_lens.append(prev_pkt_len)

  #extend dataframe with the two new columns
  new_df["time_lag"]=time_lags
  new_df["prev_pkt_len"]=prev_pkt_lens

  logger.log("Adding time lags and prev_pkt_lens...",logger.OK)

  #### Now, let's add the prev_time_lag column
  df_temp = new_df[["src_ip", "dst_ip", "src_port", "dst_port", "time_lag"]]
  current_id = 0 #reset for progress tracking
  
  
  flows = dict()
  prev_time_lags = list()
  current_id = 0 #reset for proper progress tracking
  for row in df_temp.itertuples(index=False):
    src_ip=row.src_ip
    dst_ip=row.dst_ip
    src_port=row.src_port
    dst_port=row.dst_port
    time_lag=row.time_lag

    current_id += 1

    logger.calculateRemainingPercentage(current_id,n, "Adding prev_time_lag...")
    ### Get/generate flowID
    flow=get_flow_id(src_ip,dst_ip,src_port,dst_port)

    if flow not in flows:
      flows[flow] = time_lag #supposed to be 0.0
      last_time_lag= time_lag #supposed to be 0.0
    else:
      last_time_lag=flows[flow]
      flows[flow] = time_lag
    
    prev_time_lags.append(last_time_lag)

  #append to dataframe
  new_df["prev_time_lag"]=prev_time_lags

  logger.log("Adding prev_time_lag...",logger.OK)
  return new_df

def balance_dataset(df):
  '''
  This function balances the dataset to have equal amount of DoH and Web packets.
  '''
  logger.log_simple("Balancing the dataset...")
  #get classes and their corresponding counts
  doh = df[df["Label"] == 1] #DoH packets
  doh_num = len(doh) #number of DoH packets
  web = df[df["Label"] == 0] #Web packets
  web_num = len(web) #number of Web packets
  sum = len(df)
  logger.log_simple("Number of packets: {}".format(sum))
  logger.log_simple("Number of DoH packets: {} ({:.2f}%)".format(doh_num, doh_num/sum*100))
  logger.log_simple("Number of Web packets: {} ({:.2f}%)".format(web_num, web_num/sum*100))
  if doh_num < web_num:
    #DoH packets are the minority
    df_minority = doh
    df_majority = web
    df_majority_num = web_num
    df_minority_num = doh_num
    logger.log_simple("DoH packets are in minority")
  else:
    #Web packets are the minority
    df_minority = web
    df_majority = doh
    df_majority_num = doh_num
    df_minority_num = web_num
    logger.log_simple("Web packets are in minority")
  
  #here we follow a downsampling strategy, but this can be easily modified
  #to be an upsampling one 
  df_resampled = resample(df_majority,
                          replace = False, #sample with replacement
                          n_samples = df_minority_num,
                          random_state=123)

  df_new = pd.concat([df_majority, df_resampled])
  logger.log("Dataset balanced")
  logger.log("Dataset balanced",logger.OK)
  return df_new

def create_dataframe():
  '''
  This function iterates through the input arguments to look for all files needed for the dataframe.
  Then, by the callback load_csvfile(), it creates the final dataframe which will be saved according to the
  DATAFRAME constant read from argument --output

  '''
  logger.log_simple("Creating dataframe...")
  dataframe = None
  no_df_yet = True
  for path,isFile in traffic_traces:
    logger.log( "Processing {}...".format(path))
    ### Actual path to process is a file, no recursion needed 
    if isFile: #if one file only, let's create a dataframe from it and merge
      df_tmp = load_csvfile(path)
      logger.log( "Processing {}...".format(path), logger.OK)
      
      #### First dataframe
      if(no_df_yet): #no dataframe has been created yet            
        dataframe = df_tmp
        no_df_yet = False #change this bit to False for the rest of the csv files
      
      #### Concatenate the new dataframe with the previous one
      else:  #dataframe already exists, we only need to merge without overwrite
        df_tmp = adjust_dataframe(df_tmp, dataframe, path)
        dataframe = pd.concat([dataframe, df_tmp])
    
    ### Actual path is a directory, we have to loop through all files
    else:  
      files = []
      for _,_,filename in walk(path): #walking through the directory
        files.extend(filename)
        break #to not go into subdirs and only list files
      
      #### Prepare files and their paths for easier processing
      tmp_files=[] #remove any files that are not compliant with the .csv files' naming patterns
      for f in files:
        if(re.search("^csvfile-[0-9]*-[0-9]*.csv",f)) is not None: #regexp for doh_docker specific csvfiles only
          tmp_files.append(path+"/"+f)
      files=tmp_files #update files list with the filtered results
      
      #### Process the new file list
      for f in files: #let's create dataframes from the .csv files
        logger.log("Processing {}...".format(f))
        #Load .csv file
        df_tmp = load_csvfile(f)
        logger.log("Processing {}...".format(f),logger.OK)

        #### First dataframe
        if(no_df_yet): #no dataframe has been created yet           
          dataframe = df_tmp
          no_df_yet = False #change this bit to False for the rest of the csv files
        
        #### Concatenate the new dataframe with the previous one
        else:  #dataframe already exists, we only need to merge without overwrite
          df_tmp = adjust_dataframe(df_tmp, dataframe, path)
          # print(dataframe.tail(1))
          # print(df_tmp.head(1))
          dataframe = pd.concat([dataframe, df_tmp]) 
        


  logger.log("Reading and parsing raw data",logger.OK)
  #reset indices
  dataframe=dataframe.reset_index(drop=True)

  ############################
  ####  data manipulation ####
  ############################

  logger.log_simple("MANIPULATING DATA",logger.TITLE_OPEN)

  # sys.stdout.write(str("Manipulating data...\r"))

  #filter out non http2 and doh packets
  logger.log("Dropping non-Web and non-DoH packets...")
  dataframe=dataframe.loc[(dataframe["Protocol"]=="HTTP2") | (dataframe["Protocol"]=="DoH")]
  logger.log("Dropping non-Web and non-DoH packets...",logger.OK)

  #replace labels with 0 (for Web) and 1 (for DoH)
  logger.log("Relabeling packets with a new column 'Label' (0 - Web, 1 - DoH)...")
  dataframe=add_label(dataframe)
  logger.log("Relabeling packets with a new column 'Label' (0 - Web, 1 - DoH)...",logger.OK)

  #add direction column
  logger.log("Adding direction to packets based on the IP...")
  dataframe=add_packet_direction(dataframe)
  logger.log("Adding direction to packets based on the IP...",logger.OK)

  #remove responses according to the direction
  logger.log("Remove response packets according to direction...")
  if not BIDIR:
    dataframe=dataframe[dataframe['direction']=='request']
    logger.log("Remove response packets according to direction...",logger.OK)
  else: 
    logger.log("Remove response packets according to direction...",logger.SKIP)

  #update labels according to whether the corresponding flow is DoH 
  dataframe=relabel_outlier_flows(dataframe)

  #add time lag 
  dataframe=add_time_lag(dataframe)

  #pkt_len and time_lag is part of the dataframe already
  #DO PADDING HERE IF NEEDED
  logger.log_simple("PADDING", logger.TITLE_OPEN)
  dataframe = pad_feature_values(dataframe, pkt_len=PAD_PKT_LEN, time_lag=PAD_TIME_LAG)
  logger.log_simple("END PADDING", logger.TITLE_CLOSE)

  #add prev_pkt_len
  dataframe=add_prev_pkt_len(dataframe)

  #add prev_time_lag
  dataframe=add_prev_time_lag(dataframe)

  # #update labels according to whether the corresponding flow is DoH 
  # dataframe=relabel_outlier_flows(dataframe)

  #balancing the dataset after all relative features are set
  if(BALANCE_DATASET):
    dataframe = balance_dataset(dataframe)
  
  logger.log_simple("MANIPULATING DATA", logger.TITLE_CLOSE)

  logger.log_simple("RANDOM DATAFRAME SNIPPET", logger.TITLE_OPEN)
  # logger.log_simple(dataframe)
  max=dataframe["flowID"].max()
  r = random.randint(0,max)
  print(dataframe[dataframe["flowID"] == r])
  logger.log_simple("END RANDOM DATAFRAME SNIPPET", logger.TITLE_CLOSE)

  logger.log("Saving dataframe to {}...".format(DATAFRAME))
  dataframe.to_pickle(DATAFRAME)
  logger.log("Saving dataframe to {}...".format(DATAFRAME),logger.OK)

  logger.log_simple("DoH DATAFRAME SNIPPET", logger.TITLE_OPEN)
  columns_of_interest=["flowID", "dst_ip", "Protocol","Label","pkt_len", "prev_pkt_len","time_lag","prev_time_lag"]
  # print(dataframe[dataframe["Label"]==1][columns_of_interest])
  print(dataframe[dataframe["Time"]> 100.0][columns_of_interest])
  logger.log_simple("END DoH DATAFRAME SNIPPET", logger.TITLE_CLOSE)

###########################################
#               MAIN                      #
###########################################

if __name__ == "__main__":
        create_dataframe()
