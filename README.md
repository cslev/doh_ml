# doh_ml
This repository is for processing the data gathered via [doh_docker](https://github.com/cslev/doh_docker) container via Python and ML techniques.
Furthermore, this repository has been made for our paper [removed for now] to appear at [removed for now].

The project constitutes several script files with different purpose. Formerly, everything was in one big script file but that was non-manageable after some point.

Let's see what you can do here.

# Lifecycle
To start playing around with the project, there is a general lifecycle we have to follow. In the beginning, we only have raw data obtained via the `doh_docker` container. Hence, the lifecycle is as follows: 
 - we create a Pandas dataframe from the raw data via `create_dataframe.py`
 - [OPTIONAL] if we want to merge different dataframes into one, we use `merge_dataframes.py`
 - to get basic DoH and Web statistical data from the dataframes generated, we use `dataframe_stats.py`
 - to build a Machine Learning model from the dataframe(s) and also test them in a closed-world setting right away, we use `train_model.py`
 - When we have models, we can evaluate any dataframe (in a closed- or open-world setting) with `test_model.py`

# Requirements
Almost everything from scikit-learn, mathplotlib, pandas, numpy and, most importantly, Python 3.

## `create_dataframe.py`
```
Create Panda dataframes for doh_ml from .csv files generated via doh_docker container

optional arguments:
  -h, --help            show this help message and exit
  -i N [N ...], --input N [N ...]
                        Specify here the path to the .csv files containing the traffictraces. 
                        Use /path/to/dir to load all .csv files within the given dir.
                        Or, use /full/path/to/csvfile-1-200.csv to load one file only.
                        Or, use multiple paths to select (and combine) multiple .csv files or whole directories. 
                        Example: -i /path/to/dir /full/path/to/csvfile-1-200.csv /path/to/dir2
  -B, --balance-dataset
                        Specify whether to balance the dataset (Default: False
  -p PAD_PKT_LEN, --pad-packet-len PAD_PKT_LEN
                        Specify whether to pad each DoH packet's pkt_len and how (Default: no padding)
                        1: Pad according to RFC 8467, i.e., to the closest multiple of 128 bytes
                        2: Pad with a random number between (1,MTU-actual packet size)
                        3: Pad to a random number from the distribution of the Web packets
                        4: Pad to a random preceding Web packet's size
                        5: Pad a sequence of DoH packets to a random sequence of preceeding Web packets' sizes
  -t PAD_TIME_LAG, --pad-time-lag PAD_TIME_LAG
                        Specify whether to pad each DoH packet's time_lag and how (Default: no padding)
                        3: Pad to a random number from the distribution of the Web packets
                        4: Pad to a random preceding Web packet's size
                        5: Pad a sequence of DoH packets to a random sequence of preceeding Web packets' sizes
  -o DATAFRAME_PATH, --output DATAFRAME_PATH
                        Specify the full path for the dataframe to be saved (Default: ./dataframes/df.pkl)
```
### Example
To create a simple dataframe from the example data provided, just issue the following command.
```
python3 create_dataframe.py -i csvfile-1-200.csv -o df.pkl
```

## `merge_dataframe.py`
```
Merge Panda dataframes for doh_ml

optional arguments:
  -h, --help            show this help message and exit
  -m N [N ...], --merge N [N ...]
                        Specify here the paths to the dataframesExample: -m /path/to/df1.pkl /path/to/df2.pkl
  -o OUTPUT, --output OUTPUT
                        Specify the full path for the dataframe to be saved (Default: ./dataframes/df.pkl)

```
### Example
```
python3 merge_dataframe.py -m df.pkl df2.pkl df3.pkl -o combined_df.pkl
```

## `dataframe_stats.py`
```
Brief analysis of dataframes created via 'create_dataframe.py'.csv files generated via doh_docker container

optional arguments:
  -h, --help            show this help message and exit
  -i DATAFRAME_PATH, --input DATAFRAME_PATH
                        Specify here the path to the dataframe
  -H, --generate-histogram
                        [TRAINING/TESTING] Specify whether to generate histograms for the datasets used for training (Default: False)
  -f N [N ...], --features N [N ...]
                        Specify the list of features to describe. 
                        Default: pkt_len, prev_pkt_len, time_lag, prev_time_lag

```
### Example
```
python3 dataframe_stats.py -i df.pkl

dataframe_stats.py -| Checking set features to be valid...                                                                                                                          [DONE]
dataframe_stats.py -| Loading from file test.df...                                                                                                                                  [DONE]
dataframe_stats.py -| +----------------------------------------------------------------------------- COLUMNS -----------------------------------------------------------------------------+
dataframe_stats.py -| Index(['No.', 'Time', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'Protocol',
       'pkt_len', 'Info', 'Label', 'direction', 'flowID', 'time_lag',
       'prev_pkt_len', 'prev_time_lag'],
      dtype='object')
dataframe_stats.py -| +=========================================================================== END COLUMNS ===========================================================================+
dataframe_stats.py -| +--------------------------------------------------------------------------- STATISTICS ---------------------------------------------------------------------------+
dataframe_stats.py -| Number of packets: 75635
dataframe_stats.py -| Number of DoH packets: 45753 (60.49%)
dataframe_stats.py -| Number of Web packets: 29882 (39.51%)
dataframe_stats.py -| +----------------------------------------------------------------------- Describing packets -----------------------------------------------------------------------+
dataframe_stats.py -|             pkt_len
count  75635.000000
mean     176.453005
std      265.028650
min       84.000000
25%      110.000000
50%      140.000000
75%      167.000000
max    20586.000000
dataframe_stats.py -|        prev_pkt_len
count  75635.000000
mean     164.383182
std      268.264882
min        0.000000
25%      110.000000
50%      138.000000
75%      158.000000
max    20586.000000
dataframe_stats.py -|            time_lag
count  75635.000000
mean       0.475347
std       21.309674
min        0.000000
25%        0.000048
50%        0.000218
75%        0.006583
max     2889.144726
dataframe_stats.py -|        prev_time_lag
count   75635.000000
mean        0.258062
std        18.491820
min         0.000000
25%         0.000040
50%         0.000150
75%         0.003568
max      2839.028146
dataframe_stats.py -| +===================================================================== End Describing packets =====================================================================+
dataframe_stats.py -| +========================================================================= END STATISITCS =========================================================================+

```

## `train_model.py`
```
This script is for training a model from DoH traces and test it right away in
a closed-world setting (with optional k-fold cross-validation). Use further
arguments to obtain more metrics.

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_DATAFRAME, --train-dataframe TRAIN_DATAFRAME
                        Specify the full path for the dataframe used for
                        training
  -m ML_MODEL_PATH, --ml-model-path ML_MODEL_PATH
                        Specify the full path for the model to save
  -f N [N ...], --features N [N ...]
                        Specify the list of features considered for training
                        (X). This should not contain the 'Label' (y). Default:
                        pkt_len, prev_pkt_len, time_lag, prev_time_lag
  -S, --generate-shapley
                        Specify whether to generate SHAPLEY values after
                        testing (Default: False)
  -P, --generate-prc    Specify whether to generate PRC curves after testing
                        (Default: False)
  -c CPU_CORE_NUM, --cpu-core-num CPU_CORE_NUM
                        Specify here the number of CPU cores to use for
                        parallel jobs (Default: 1)
  -C CROSS_VALIDATION, --cross-validation CROSS_VALIDATION
                        Specify here if K-fold cross-validation is needed with
                        the number of fold you want to use (Default: No cross
                        validation
```
### Example
```
python3 train_model.py -t test.df -m test_model.pkl -c 20

...
train_model.py -| Accuracy : 0.9978847170809095
train_model.py -| Precision: 0.9967061923583662
train_model.py -| Recall:    0.9997797356828194
train_model.py -| F1 Score:  0.9982405981966131
train_model.py -| Confusion Matrix :
[[3009   15]
 [   1 4539]]
train_model.py -| Again, the features used for training: ['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_time_lag']
train_model.py -| +======================================================================== END CLOSED WORLD SETTING ========================================================================+
train_model.py -| Saving the model to test_model.pkl                                                                                                                                [DONE]

```

## `test_model.py`
```
This script is for loading a trained model from a file and test it with a
dataframe (passed as argument) in an open-world setting. Use further arguments
to obtain more metrics.

optional arguments:
  -h, --help            show this help message and exit
  -m ML_MODEL_PATH, --ml-model-path ML_MODEL_PATH
                        Specify the full path for the model to load
  -t TEST_DATAFRAME, --test-dataframe TEST_DATAFRAME
                        Specify the full path for the dataframe used for
                        testing
  -o OUTPUT, --output OUTPUT
                        Specify output basename used for PRC, shapley, etc.
  -f N [N ...], --features N [N ...]
                        Specify the list of features considered for testing
                        (X). This should not contain the 'Label' (y). Default:
                        pkt_len, prev_pkt_len, time_lag, prev_time_lag
  -S, --generate-shapley
                        Specify whether to generate SHAPLEY values after
                        testing (Default: False)
  -P, --generate-prc    Specify whether to generate PRC curves after testing
                        (Default: False)
  -A, --generate-roc-auc
                        Specify whether to generate ROC curves after testing
                        (Default: False)
  -c CPU_CORE_NUM, --cpu-core-num CPU_CORE_NUM
                        Specify here the number of CPU cores to use for
                        parallel jobs (Default: 1)
```
### Example
```
python3 test_model.py -m test_model.pkl -t test.df -c 20 -o test
...
test_model.py -| Testing model...                                                                                                                                                   [DONE]
test_model.py -| Accuracy : 0.9977391419316454
test_model.py -| Precision: 0.9962980706415225
test_model.py -| Recall:    0.9999781435097153
test_model.py -| F1 Score:  0.9981347150259068
test_model.py -| Confusion Matrix :
[[29712   170]
 [    1 45752]]
test_model.py -| Again, the features used for training: ['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_time_lag']
...

```
As you can see, in the latter example, we were actually using the same dataframe that we used for training, therefore it is still a closed-world setting.
The slight differences in the accuracy metrics are from the fact that in this case, the dataframe is fully used, i.e., 90% of the data points are the same used for training. So, in those points, the model is 100% sure, only the rest is different.


# How to get more csv files?
Check out [https://github.com/cslev/doh_docker](https://github.com/cslev/doh_docker)



