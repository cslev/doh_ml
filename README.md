```
python3 ./train_and_test.py -C ./data/google -d google_df.pkl
```
# doh_ml
This repository is for processing the data of doh_docker container via Python and ML techniques

# Usage
```
usage: train_and_test.py [-h] [-r TRAFFIC_TRACES] [-D] [-d DATAFRAME_PATH]
                         [-M] [-m ML_MODEL_PATH] [-p PAD_PKT_LEN] [-H] [-S]
                         [-P] [-t PAD_TIME_LAG] [-o OPEN_WORLD_DATAFRAME]
                         [-O OPEN_WORLD_RAW_DATA] [-C] [-y OUTPUT_DIR]
                         [-c CPU_CORE_NUM] [-j OVERRIDE_META]

Train and test ML model from DoH traces

optional arguments:
  -h, --help            show this help message and exit
  -r TRAFFIC_TRACES, --raw-traffic-traces TRAFFIC_TRACES
                        [TRAINING] Specify here the traffic traces located in
                        /mnt/storage/Doh_traces separated by commas for
                        training. In other words, the traffic traces (i.e.,
                        their .csv files) defined here will be iteratively
                        read and used for training one model (Default:
                        cloudflare, i.e., only cloudflare data will be used
                        for training)
  -D, --load-dataframe  [TRAINING/TESTING]Specify whether dataframe should be
                        loaded from file(Default: False, i.e. generate from
                        raw data)
  -d DATAFRAME_PATH, --dataframe-path DATAFRAME_PATH
                        [TRAINING/TESTING]Specify the full path for the
                        dataframe to load/save (Default: ./train_df.pkl)
  -M, --load-ml-model   [TRAINING/TESTING] Specify whether the ml-model should
                        be loaded from file (Default:False, i.e., trains a
                        model from scratch)
  -m ML_MODEL_PATH, --ml-model-path ML_MODEL_PATH
                        [TRAINING/TESTING] Specify the full path for the model
                        to load/save (Default: None, i.e., model will not be
                        saved)
  -p PAD_PKT_LEN, --pad-packet-len PAD_PKT_LEN
                        [TRAINING/TESTING] Specify whether to pad each DoH
                        packet's pkt_len and how (Default: no padding)1: Pad
                        according to RFC 8467, i.e., to the closest multiple
                        of 128 bytes.2: Pad with a random number between
                        (1,MTU-actual packet size)3: Pad to a random number
                        from the distribution of the Web packets4: Pad to a
                        random preceding Web packet's size5: Pad a sequence of
                        DoH packets to a random sequence of preceeding Web
                        packets' sizes
  -H, --generate-histogram
                        [TRAINING/TESTING] Specify whether to generate
                        histograms for the datasets used for training
                        (Default: False)
  -S, --generate-shapley
                        [TRAINING/TESTING] Specify whether to generate SHAPLEY
                        values after testing (Default: False)
  -P, --generate-prc    [TRAINING/TESTING] Specify whether to generate PRC
                        curves after testing (Default: False)
  -t PAD_TIME_LAG, --pad-time-lag PAD_TIME_LAG
                        [TRAINING/TESTING] Specify whether to pad each DoH
                        packet's time_lag and how (Default: no padding)3: Pad
                        to a random number from the distribution of the Web
                        packets4: Pad to a random preceding Web packet's
                        size5: Pad a sequence of DoH packets to a random
                        sequence of preceeding Web packets' sizes
  -o OPEN_WORLD_DATAFRAME, --open-world-dataframe OPEN_WORLD_DATAFRAME
                        [TESTING] Open-world setting: Specify here the path to
                        the data frame you want to use for testing. If no
                        dataframe is avalilable, use -u setting instead to
                        point to the raw files! (Default:None, i.e. Closed-
                        world setting)
  -O OPEN_WORLD_RAW_DATA, --open-world-raw-data OPEN_WORLD_RAW_DATA
                        [TESTING] Open-world setting: Specify here the path to
                        the raw data to use for testing. In this case, the raw
                        csv files will be used under the directory specified
                        via this parameter to build the data frame first. If
                        dataframe is avalilable, use -o setting instead!
                        (Default: None, i.e. Closed-world setting)
  -C, --create-dataframe-only
                        [MISC] Set if dataframe creation and saving is needed
                        only! Use only with -d/--dataframe-path to set where
                        to store the dataframe (Default: False)
  -y OUTPUT_DIR, --output-dir OUTPUT_DIR
                        [MISC] Output directory where histograms, shapley
                        values, PRC curves and accuracy metrics will be saved.
                        (Default: .)
  -c CPU_CORE_NUM, --cpu-core-num CPU_CORE_NUM
                        [MISC] Specify here the number of CPU cores to use for
                        parallel jobs (Default: 1)
  -j OVERRIDE_META, --override-meta OVERRIDE_META
                        [MISC] Specify here any extra metadata (Default: None)
```

# Examples

## Closed-world example
Create a dataframe and model from a dataset, and evalute it within a closed-world setting

In order to process the raw `.csv` files for DoH resolver `cloudflare` located under `./data/cloudflare` directory by using `4` CPU cores, 
issue the following command. 
This will also save the created dataframe and the ML model as `cloudflare_df.pkl` and as cloudflare_model.pkl, respectively.
Besides, it create histogram from the data, SHAP values for the model, and also creates PRC curve.

```
python3 ./train_and_test.py -R ./data/ -r cloudflare -d cloudflare_df.pkl -m cloudflare_model.pkl -H -S -P -c 4 
```


## Open-world example
Evaluate a created model within an open-world setting (using `4` CPU cores), 
where the data for the latter (e.g., stored as `google`) is still in raw .csv files, however, the model we use is already built.


```
python3 ./train_and_test.py -M -m cloudflare_model.pkl -c 4 -O ./data/google 
```


We can create only a dataframe first from the raw data via:
```
python3 ./train_and_test.py -C  -d google_df.pkl -R ./data -r google
```

Use -r with a list to create dataframes according to more than one raw dataset.
```
python3 ./train_and_test.py -C  -d my_openworld_df.pkl -R ./data -r google,cleanbrowsing,quad9
```


Then, use the dataframe instead for the open-world evaluations"
```
python3 ./train_and_test.py -M -m cloudflare_model.pkl -c 4 -o my_openworld_df.pkl
```

# How to get data?
Check out [https://github.com/cslev/doh_docker](https://github.com/cslev/doh_docker)



