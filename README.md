# doh_ml
This repository is for processing the data gathered via [doh_docker](https://github.com/cslev/doh_docker) container via Python and ML techniques.

The project constitutes several script files with different purpose. Formerly, everything was in one big script file but that was non-manageable after some point.

Let's see what you can do here.

# Dissemination
This repository has also been made for our [research paper](https://github.com/cslev/doh_ml/raw/main/DNS_over_HTTPS_identification.pdf) titled **Privacy of DNS-over-HTTPS: Requiem for a Dream?** to appear at [IEEE Euro S&P](http://www.ieee-security.org/TC/EuroSP2021/).

When using the repo, please use the full reference to our paper as follows:
```
@inproceedings{doh_identification_ml,
 author = {Levente Csikor and Himanshu Singh and Min Suk Kang and Dinil Mon Divakaran},
 title = {{Privacy of DNS-over-HTTPS: Requiem for a Dream?}},
 booktitle = {IEEE Euro Security and Privacy},
 year = {2021}

} 
```

# Lifecycle
To start playing around with the project, there is a general lifecycle we have to follow. In the beginning, we only have raw data obtained via the `doh_docker` container. Hence, the lifecycle is as follows: 
 - we create a Pandas dataframe from the raw data via `create_dataframe.py`
 - [OPTIONAL] if we want to merge different dataframes into one, we use `merge_dataframes.py`
 - to get basic DoH and Web statistical data from the dataframes generated, we use `dataframe_stats.py`
 - to build a Machine Learning model from the dataframe(s) and also test them in a closed-world setting right away, we use `train_model.py`
 - When we have models, we can evaluate any dataframe (in a closed- or open-world setting) with `test_model.py`

# Requirements
Almost everything from scikit-learn, mathplotlib, pandas, numpy and, most importantly, Python 3.

# Tools/scripts provided
## `create_dataframe.py`
This is one of the main tools that creates the corresponding Pandas dataframes for training, testing, and much more.
You only have to fed this script with the directory of the .csv files or the files/directories one-by-one...as you wish.
Every final feature is based on flows, but you don't have to do anything with this, the script does it automatically.

For instance, every flow's packets will be automatically relabelled to DoH if there was at least one DoH packet within the same flow. 
Accordingly, you _do not_ have to specify any IP address or whatnot to indicate this...just leave it to this guy.

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
  -b, --bidir           Specify if dataframe should be bidirectional. 
                        Default: False (only requests will be present
  -o DATAFRAME_PATH, --output DATAFRAME_PATH
                        Specify the full path for the dataframe to be saved (Default: ./dataframes/df.pkl)
```
### Example
To create a simple dataframe from the example data provided, just issue the following command.
```
python3 create_dataframe.py -i csvfile-1-200.csv -o df.pkl
```
### Troubleshooting
It can happen that the .csv files gathered via the [containers](https://github.com/cslev/doh_docker) contain malformed data. In this case, the `create_dataframe.py` scripts raises an error like this:
```
Traceback (most recent call last): data_raw/SG/comcast/csvfile-3801-4000.csv...
  File "create_dataframe.py", line 1254, in <module>
    create_dataframe()
  File "create_dataframe.py", line 1153, in create_dataframe
    df_tmp = load_csvfile(f)
  File "create_dataframe.py", line 566, in load_csvfile
    df = pd.read_csv(filename, header=0, names=list(rename_table.keys()))
  File "/usr/lib/python3/dist-packages/pandas/io/parsers.py", line 678, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/usr/lib/python3/dist-packages/pandas/io/parsers.py", line 446, in _read
    data = parser.read(nrows)
  File "/usr/lib/python3/dist-packages/pandas/io/parsers.py", line 1036, in read
    ret = self._engine.read(nrows)
  File "/usr/lib/python3/dist-packages/pandas/io/parsers.py", line 1848, in read
    data = self._reader.read(nrows)
  File "pandas/_libs/parsers.pyx", line 876, in pandas._libs.parsers.TextReader.read
  File "pandas/_libs/parsers.pyx", line 891, in pandas._libs.parsers.TextReader._read_low_memory
  File "pandas/_libs/parsers.pyx", line 945, in pandas._libs.parsers.TextReader._read_rows
  File "pandas/_libs/parsers.pyx", line 932, in pandas._libs.parsers.TextReader._tokenize_rows
  File "pandas/_libs/parsers.pyx", line 2112, in pandas._libs.parsers.raise_parser_error
pandas.errors.ParserError: Error tokenizing data. C error: Expected 9 fields in line 85750, saw 10
```
This actually happens because in some payloads there are non-UTF-8 characters and also quite random double-quotes (") that breaks the dataframe creation.
The easiest way to resolve this issue is to remove those packets from the .csv files manually. They are anyway for the _response_ packets only.

An automated way to do this is to look for the pattern of the malformed payloads. 

They look like this:
```
"447505","3016.434045","14.0.41.112","172.25.0.2","443","35290","HTTP2","1434","HEADERS[29]: 200 OK, DATA[29]Standard query 0xffe0 URI <Root> Unknown (65499) <Root> Unknown (2054) <Root> Unknown (529) \005\b\a\a\a\t.\b\n\f\024\r\f\v\v\f.\022\023\017\024\035\032\037\036\035\032\034\034 $.' ",#\034\034(7).01444\037'9=82<.342��\000C\001\t\t\t\f\v\f\030\r\r\0302!\034!222222222.2222222222222222222222222222222222222222��\000\021\b\000P\000�\003." Unused \001��\000\037\000\000\001\005\001\001\001\001\001\001\000\000 Unknown (258) <Root> Unknown (22617) <Unknown extended label>[Malformed Packet]"

or

"163470","1116.779482","23.60.171.249","172.25.0.2","443","38722","DNS","1431","Unknown operation (9) 0x4632 Unused <Root> Unused \000\0000�\000\0023�\000\000\000\000\000\000\000\000\000\000\000\000 Unused <Root>[Malformed Packet], HEADERS[65]: 200 OK, DATA[65]"
```
So, the pattern is `<Root> Un`. 

We have to remove these lines from the file, store them in new .csv files and then rename the new files back to the old ones as the script requires them to have the original naming convention. 

To do this, you can do this:
```
cd PATH/TO/CSV/FILES
for i in $(ls *csv);do echo $i; wc -l $i;cat $i|grep -v "<Root> Un" > "${i}_new.csv"; wc -l "${i}_new.csv";rm -rf $i; mv ${i}_new.csv $i;done
```
This hacky script will also shows how many lines have been removed. More precisely, it shows you the line counts for all files, and you can see how many lines have been removed.

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
This is for generating stats for the dataframes. The below script can also make boxplots for the DoH and Web packets if requested to see their statistical data in a graphical way.
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
  -b, --bidir           Specify if dataframe is bidirectional. 
                        Default: False
  -B, --boxplot         Specify if boxplots are needed. 
                        Default: False
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
  -o OUTPUT, --output OUTPUT
                        Specify output dir for PRC, shapley, etc.

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
test_model.py -| Again, the features used for testing: ['pkt_len', 'prev_pkt_len', 'time_lag', 'prev_time_lag']
...

```
As you can see, in the latter example, we were actually using the same dataframe that we used for training, therefore it is still a closed-world setting.
The slight differences in the accuracy metrics are from the fact that in this case, the dataframe is fully used, i.e., 90% of the data points are the same used for training. So, in those points, the model is 100% sure, only the rest is different.


# How to get more csv files?
Check out [https://github.com/cslev/doh_docker](https://github.com/cslev/doh_docker)



