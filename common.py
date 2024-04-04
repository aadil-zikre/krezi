from pickle import TRUE
import __main__

# if something in dir(__main__):

# from IPython.display import display, HTML
import IPython.display as ip_display
def fit_to_screen():
    ip_display.display(ip_display.HTML("<style>.container { width:100% !important; }</style>"))
fit_to_screen()

import numpy as np
import pandas as pd
import datetime 
import time 
import itables
import logging
from file_logger import Logger
import importlib
import sys
import gc
import pickle 
from mp_util import run_in_parallel

data_dir = "/home/azikre/Python/notebooks/data/"
log_dir = "/home/azikre/Python/notebooks/logs/"
tmp_dir = "/home/azikre/Python/notebooks/tmp/"

import types
def imported_modules():
    for name, val in globals().items():  # Everytime Globals is called, dict size will change as every run adds a new cell run info like _i120 and _120
        if isinstance(val, types.ModuleType):
            yield val.__name__, name

def display_imported_modules():
    for module, alias in imported_modules(): print(f"{module:<50} as {' '*5}{alias}")

def elapsed_time(start_tm):
    et = time.time() - start_tm
    if et < 120 : 
        return f"{et:.1f} seconds"
    else : 
        return f"{et/60:.1f} minutes"

# ^Decorator
def func_et(func):
    def func_wrapper(*args, **kwargs):
        log_info(f"{func.__name__} :: Function Execution Started")
        start_tm = time.time()
        res = func(*args, **kwargs)
        time_taken = elapsed_time(start_tm)
        log_info(f"{func.__name__} :: Function Execution Done :: Time Taken -> {time_taken}")
        if res: return res
    return func_wrapper

def custom_round_nearest_p5(number):
    x = number%1
    if x==0:
        return number
    elif x<=0.5:
        return int(number)+0.5
    elif x>0.5:
        return int(number)+1.0

STARTING_THRESHOLD = 0
def accuracy_calculator_xgb_reg(y_actual, y_pred, apply_round = False, round_engine = custom_round_nearest_p5, thresh = STARTING_THRESHOLD):
    """y_actual -> actual, y_pred -> prediction"""
    if apply_round:
        y_actual = pd.Series(y_actual).apply(round_engine).reset_index(drop = True)
        y_pred = pd.Series(y_pred).apply(round_engine).reset_index(drop = True)
    
    non_zero_index = y_actual>0
    
    y_actual = y_actual[non_zero_index]
    y_pred = y_pred[non_zero_index]
    
    if len(y_actual)!=len(y_pred):
        raise ValueError(f"Length of y_actual and y_pred must be same. Please check and try again. Found {len(y_actual)} & {len(y_pred)}")
    
    acc_mape = 100 - np.mean(np.abs((y_actual-y_pred)/y_actual*100))
    acc_mae = np.mean(np.abs(y_actual-y_pred))
    
    if len(y_actual) > 0:
        acc_10_perc = sum(((y_pred >= y_actual * 0.9) & (y_pred <= y_actual * 1.1)) | (np.abs(y_pred-y_actual) <=thresh)) / len(y_actual)*100
        acc_15_perc = sum(((y_pred >= y_actual * 0.85) & (y_pred <= y_actual * 1.15)) | (np.abs(y_pred-y_actual) <=thresh)) / len(y_actual)*100
        acc_20_perc = sum(((y_pred >= y_actual * 0.8) & (y_pred <= y_actual * 1.2)) | (np.abs(y_pred-y_actual) <=thresh)) / len(y_actual)*100
        acc_30_perc = sum(((y_pred >= y_actual * 0.7) & (y_pred <= y_actual * 1.3)) | (np.abs(y_pred-y_actual) <=thresh)) / len(y_actual)*100
        perc_op = sum((y_pred >= y_actual ))/len(y_actual)*100
        perc_up = sum((y_pred < y_actual ))/len(y_actual)*100
    else:
        acc_10_perc=acc_15_perc=acc_20_perc=perc_op=perc_up = np.nan
    
    return {"ACCURACY" : round(acc_mape,2),
            "MAPE" :round(100-acc_mape,2),
            "MAE" : round(acc_mae,2),
            "ACC_10%" : round(acc_10_perc,2),
#             "ACC_15%" : round(acc_15_perc,2),
            "ACC_20%" : round(acc_20_perc,2),
            "ACC_30%" : round(acc_30_perc,2),
            "over_predicted%" : round(perc_op,2), 
            "under_predicted%" : round(perc_up,2)}

def create_date_features(df,date_col):
    # Date Features
    df[date_col] = pd.to_datetime(df[date_col])
    df['date'] = df[date_col]
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['weekofyear'] = df.date.dt.weekofyear
    df['is_weekend'] = [1 if x.dayofweek in [5,6] else 0 for x in df['date']]
    df['daysinmonth'] = df.date.dt.daysinmonth
    
#     checks for holidays
    #public_holiday_list = set(holidays[holidays['holiday']=='public'].ds)
    #not_pulic_holiday_list = set(holidays[holidays['holiday']!='public'].ds)
    #df['is_public_holiday'] = df["date"].apply(lambda x: 1 if x in public_holiday_list else 0)
    #df['is_not_public_holiday'] = df["date"].apply(lambda x: 1 if x in not_pulic_holiday_list else 0)
    # holiday_list=set(holidays.ds)
   # df['is_public_holiday'] = df["date"].apply(lambda x: 1 if x in public_holiday_list else 0)
   # df['is_not_public_holiday'] = df["date"].apply(lambda x: 1 if x in not_pulic_holiday_list else 0)
    # df['is_holiday']=df["date"].apply(lambda x: 1 if x in holiday_list else 0)
    
#     df["dotd_active"] = df["date"].apply(lambda x: 1 if x in list(df_dotd.date) else 0)
    
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype(int)
    df['is_year_start'] = df.date.dt.is_year_start.astype(int)
    df['is_year_end'] = df.date.dt.is_year_end.astype(int)
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = np.where(df.month.isin([9, 10, 11]), 3, df["season"])
    
    # Additionnal Data Features
    df['day^year'] = np.log((np.log(df['dayofyear'] + 1)) ** (df['year'] - 2000))
    
    # Drop date
    df.drop('date', axis=1, inplace=True)
    
    return df

def flatten_list_of_lists(t):
    return [item for sublist in t for item in sublist]

from tqdm.notebook import tqdm_notebook
def tqdm_custom(iterable,**add_params):
    bar_format = "{desc}: {percentage:.3f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_inv_fmt}"
    return tqdm_notebook(iterable,bar_format = bar_format,**add_params)

def to_map(self, key_col, value_col):
    return self.set_index(key_col)[value_col].to_dict()
pd.DataFrame.to_map = to_map

def read_parquet_from_s3(s3_path, FILENAME):
    this_file_name = FILENAME

    filename = this_file_name if FILENAME.endswith(".parquet") else f'{this_file_name}.parquet'
    s3_dir = 's3://analytics.faasos.io'
    s3_path = f'{s3_dir}/{s3_path}'
    s3_path = s3_path if s3_path.endswith("/") else s3_path+"/"

    cmd(f"aws s3 cp {s3_path}{filename} /tmp/")

    return pd.read_parquet(f'/tmp/{filename}')

def to_parquet_to_s3(df, s3_path, FILENAME):
    this_file_name = FILENAME

    filename = this_file_name if FILENAME.endswith(".parquet") else f'{this_file_name}.parquet'
    df.to_parquet(f'/tmp/{filename}', index=False)
    
    s3_dir = 's3://analytics.faasos.io'
    s3_path = f'{s3_dir}/{s3_path}'
    s3_path = s3_path if s3_path.endswith("/") else s3_path+"/"

    cmd(f"aws s3 cp /tmp/{filename} {s3_path}")

# command = f"aws s3 ls {s3_dir} --human-readable --summarize | sort -k 2 -r | awk '{{print $5}}'"
import subprocess
def run_command(command, return_output = False):
    with subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, universal_newlines=True) as p:
        print(f"Running Command :: {command}")
        output = p.stdout.readlines()
        print("output :: ", ''.join(output))
        print("error :: ", p.stderr.read())
    print("return code :: ", p.returncode)
    if return_output:
            res = set(output)
            res.discard('\n')
            return list(res)
cmd = run_command

def run_command_return_output(command):
    return run_command(command, return_output = True)

logger = Logger('acf')
log = logger.get_logger() # You can change the date format here

log_debug = log.debug
log_info = log.info
log_warning = log.warning
log_error = log.error
log_critical = log.critical


log_info("Logger initialized WITHOUT file handler")

def init_logger(filepath):
    global log
    log = logger.add_file_handler(filepath)
    log_info(f"File Handler added. Location set to {filepath!r}")

## Printing Dataframes as DataTables 
# https://github.com/mwouts/itables/blob/61c1c916175a77f27623eb93fa4ecf42b9a7b7b4/itables/options.py
# for options

pd.DataFrame.data_table = itables.show
pd.DataFrame.dT = pd.DataFrame.data_table

def reload_modules(modules:list) -> None:
    for module in modules:
        importlib.reload(module)


def apply_on_series(df, func):
    return df.progress_apply(func)

def multicore_apply_by_chunks(df_st, func, max_chunks, max_workers = None, log_progress=False):
    max_workers = max_chunks if not max_workers else max_workers
    len_df = df_st.shape[0]
    n = len_df // max_chunks +1
    chunks = [df_st[i:i+n] for i in range(0,len_df,n)]
    fn_specs = []
    for i in chunks:
        fn_specs.append((apply_on_series, {"df" : i, 'func': func}))
    
    results, exceptions = run_in_parallel(fn_specs, threaded = False, max_workers = max_workers, log_at = 1, log_func = log_info, log_progress = log_progress)
    if len(exceptions) > 0 : print(exceptions[0][2])
    results = pd.concat(i[1] for i in results)
    del chunks
    del fn_specs
    del exceptions
    gc.collect()
    return results

pd.Series.multicore_apply_by_chunks = multicore_apply_by_chunks

def to_pickle(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def read_pickle(path):
    with open(path, 'rb') as f:
        object = pickle.load(f)
    return object

def custom_mode(arr):
    freq = defaultdict(int)
    for i in arr:
        freq[i] += 1
    max_freq = max(freq.values())
    max_at = [k for k,v in freq.items() if v == max_freq]
    if len(max_at) > 1:
        return -1
    else:
        return max_at[0]

def custom_mode_return_all(arr):
    freq = defaultdict(int)
    for i in arr:
        freq[i] += 1
    max_freq = max(freq.values())
    max_at = [k for k,v in freq.items() if v == max_freq]
    return max_at

import ctypes
def get_object_at_address(address):
    '''
    address=0x7822e7c7bac0
    print(f"Value from address {address} is {ctypes.cast(address, ctypes.py_object).value}")
    '''
    obj = ctypes.cast(address, ctypes.py_object).value
    return obj