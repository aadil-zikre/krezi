from pickle import TRUE
import __main__

# if something in dir(__main__):

# from IPython.display import display, HTML
import IPython.display as ip_display
def fit_to_screen():
    """
    A function to fit the display to the screen by modifying the HTML style.
    """
    ip_display.display(ip_display.HTML("<style>.container { width:100% !important; }</style>"))
fit_to_screen()

import numpy as np
import pandas as pd
import datetime 
import time 
import itables
import logging
from krezi.logging_util.file_logger import Logger
import importlib
import sys
import os
import gc
import pickle
import functools 
from collections import defaultdict
from krezi.multiprocessing_util.mp_util import run_in_parallel

home_dir = os.environ.get('HOME_DIR') 
notebooks_dir = os.environ.get('NOTEBOOKS_DIR')
data_dir = os.environ.get('DATA_DIR') 
log_dir = os.environ.get('LOG_DIR') 
tmp_dir = os.environ.get('TMP_DIR')

import types
def imported_modules():
    """
    Generates a generator that yields the names and values of imported modules.

    This function iterates over the global namespace and checks if each value is an instance of the `types.ModuleType` class. If it is, the function yields a tuple containing the module's name and the corresponding variable name.

    Yields:
        tuple: A tuple containing the module name (str) and the variable name (str) of an imported module.

    Example:
        >>> for module_name, var_name in imported_modules():
        ...     print(module_name, var_name)
        module1 var1
        module2 var2
    """
    for name, val in globals().items():  # Everytime Globals is called, dict size will change as every run adds a new cell run info like _i120 and _120
        if isinstance(val, types.ModuleType):
            yield val.__name__, name

def display_imported_modules():
    """
    Display the imported modules and their aliases.

    This function iterates over the imported modules and their aliases using the `imported_modules` function. It prints the module name followed by its alias, with a fixed width of 50 characters for the module name and 5 spaces for the alias.

    Parameters:
        None

    Returns:
        None
    """
    for module, alias in imported_modules(): 
        print(f"{module:<50} as {' '*5}{alias}")

def elapsed_time(start_tm):
    """
    Calculate the elapsed time from a given start time.

    Args:
        start_tm (float): The start time in seconds since the epoch.

    Returns:
        str: The elapsed time in seconds if it is less than 120 seconds, otherwise in minutes.
    """
    et = time.time() - start_tm
    if et < 120 : 
        return f"{et:.1f} seconds"
    else : 
        return f"{et/60:.1f} minutes"

# ^Decorator
def func_et(func):
    """
    Decorator function that logs the start and end of a function execution, as well as the time taken to execute it.

    Parameters:
        func (function): The function to be wrapped.

    Returns:
        function: The wrapped function.

    Example:
        @func_et
        def my_function():
            # Function code here
            pass

        # Output:
        # log_prefix :: my_function :: Function Execution Started
        # log_prefix :: my_function :: Function Execution Done :: Time Taken -> 0.123 seconds
    """
    @functools.wraps
    def func_wrapper(*args, **kwargs):
        log_info(f"{func.__name__} :: Function Execution Started")
        start_tm = time.time()
        res = func(*args, **kwargs)
        time_taken = elapsed_time(start_tm)
        log_info(f"{func.__name__} :: Function Execution Done :: Time Taken -> {time_taken}")
        if res: return res
    return func_wrapper

def custom_round_nearest_p5(number):
    """
    Rounds a number to the nearest multiple of 0.5.

    Parameters:
        number (float): The number to be rounded.

    Returns:
        float: The rounded number.

    Example:
        >>> custom_round_nearest_p5(2.3)
        2.5
        >>> custom_round_nearest_p5(3.7)
        4.0
        >>> custom_round_nearest_p5(0.0)
        0.0
    """
    x = number%1
    if x==0:
        return number
    elif x<=0.5:
        return int(number)+0.5
    elif x>0.5:
        return int(number)+1.0

STARTING_THRESHOLD = 0
def accuracy_calculator_xgb_reg(y_actual, y_pred, apply_round = False, round_engine = custom_round_nearest_p5, thresh = STARTING_THRESHOLD):
    """
    Calculates the accuracy of a regression model using XGBoost.
    
    Args:
        y_actual (array-like): The actual values.
        y_pred (array-like): The predicted values.
        apply_round (bool, optional): Whether to apply rounding to the actual and predicted values. Defaults to False.
        round_engine (function, optional): The rounding function to apply. Defaults to custom_round_nearest_p5.
        thresh (float, optional): The threshold value for calculating the accuracy. Defaults to STARTING_THRESHOLD.
        
    Returns:
        dict: A dictionary containing the accuracy metrics:
            - "ACCURACY" (float): The accuracy in percentage.
            - "MAPE" (float): The mean absolute percentage error.
            - "MAE" (float): The mean absolute error.
            - "ACC_10%" (float): The accuracy at 10% tolerance level.
            - "ACC_15%" (float): The accuracy at 15% tolerance level.
            - "ACC_20%" (float): The accuracy at 20% tolerance level.
            - "ACC_30%" (float): The accuracy at 30% tolerance level.
            - "over_predicted%" (float): The percentage of over-predicted values.
            - "under_predicted%" (float): The percentage of under-predicted values.
    """
    
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
            "ACC_15%" : round(acc_15_perc,2),
            "ACC_20%" : round(acc_20_perc,2),
            "ACC_30%" : round(acc_30_perc,2),
            "over_predicted%" : round(perc_op,2), 
            "under_predicted%" : round(perc_up,2)}

def create_date_features(df, date_col, holidays = None):
    """
    Creates date features for a given DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the date column.
        date_col (str): The name of the date column.
        holidays (pandas.DataFrame, optional): A DataFrame containing holiday information. Defaults to None.

    Returns:
        pandas.DataFrame: The DataFrame with additional date features.

    Raises:
        None
    """
    # * Date Features
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
    
    # * checks for holidays
    if holidays:
        public_holiday_list = set(holidays[holidays['holiday']=='public'].ds)
        not_pulic_holiday_list = set(holidays[holidays['holiday']!='public'].ds)
    
        df['is_public_holiday'] = df["date"].apply(lambda x: 1 if x in public_holiday_list else 0)
        df['is_not_public_holiday'] = df["date"].apply(lambda x: 1 if x in not_pulic_holiday_list else 0)
    
    # * start and end features
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype(int)
    df['is_year_start'] = df.date.dt.is_year_start.astype(int)
    df['is_year_end'] = df.date.dt.is_year_end.astype(int)
    
    # * Season
    # TODO: As seasons dont necessarily have a hierarchy, consider adding is_<_season_> features
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df['season'] = 1
    df.loc[df['month'].isin([12, 1, 2]), 'season'] = 0
    df.loc[df['month'].isin([6, 7, 8]), 'season'] = 2
    df.loc[df['month'].isin([9, 10, 11]), 'season'] = 3
    
    # * Date Year Combination
    df['day^year'] = np.log((np.log(df['dayofyear'] + 1)) ** (df['year'] - 2000))
    
    return df

pd.DataFrame.create_date_features = create_date_features

def flatten_list_of_lists(t):
    """
    Flattens a list of lists into a single list.

    Args:
        t (list): A list of lists to be flattened.

    ALternative:
        Use itertool's chain method.

    Returns:
        list: A flattened list containing all the elements from the input list of lists.
    """
    return [item for sublist in t for item in sublist]

from tqdm.notebook import tqdm_notebook
def tqdm_custom(iterable,**add_params):
    """
    Creates a custom tqdm progress bar for iterating over an iterable.

    Args:
        iterable (iterable): The iterable to iterate over.
        **add_params (dict): Additional parameters to pass to tqdm_notebook.

    Returns:
        tqdm.notebook.tqdm_notebook: A tqdm progress bar for iterating over the iterable.

    Note:
        The custom bar format includes the description, percentage, bar, current and total item counts, elapsed time, and remaining time.
    """
    bar_format = "{desc}: {percentage:.3f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_inv_fmt}"
    return tqdm_notebook(iterable,bar_format = bar_format,**add_params)

def to_map(self, key_col, value_col):
    """
    Generates a dictionary mapping values from the specified columns in the DataFrame.

    Parameters:
        key_col (str): The name of the column to be used as keys in the dictionary.
        value_col (str): The name of the column to be used as values in the dictionary.

    Returns:
        dict: A dictionary mapping values from the key_col to values from the value_col.
    """
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
    """
    Reloads the specified modules.

    Args:
        modules (list): A list of modules to reload.

    Returns:
        None
    """
    for module in modules:
        importlib.reload(module)

def reload_krezi():
    """
    Reloads the 'krezi' module.

    This function reloads the 'krezi' module by calling the 'reload_modules' function with the list containing the 'krezi' module as its argument.

    Parameters:
        None

    Returns:
        None
    """
    reload_modules(['krezi'])

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
    """
    Returns a list of all elements in the input array that have the highest frequency.
    
    Parameters:
        arr (Iterable): An iterable of elements.
        
    Returns:
        List[Any]: A list of elements that have the highest frequency in the input array.
    """
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