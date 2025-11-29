from pathlib import Path
import polars as pl

from typing import Optional

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from datetime import datetime


def _drop_nulls(df: pl.DataFrame, df_name: str, col_names: Optional[list[str]] = None) -> pl.DataFrame:
    """ Drop null rows in df with logging

    Args:
        df (pl.DataFrame): 
        df_name (str): 
        col_names (Optional[list[str]], optional): select columns to check for nulls. Defaults to None.

    Returns:
        pl.DataFrame: _description_
    """
    len_df = len(df)
    df = df.drop_nulls(subset=col_names)
    len_dropped_df = len(df)
    if len_df != len_dropped_df:
        logger.warning(f"Dropped {len_df - len_dropped_df} null rows from {df_name} dataframe")
    return df

    
def _filter_on_timestamp(df: pl.DataFrame,
                         df_name: str,
                         ts_col: str,
                         min_ts: Optional[datetime] = None,
                         max_ts: Optional[datetime] = None) -> pl.DataFrame:
    """ Remove rows with timestamps before min_ts or after max_ts

    Args:
        df (pl.DataFrame): 
        df_name (str): (for logging purposes)
        ts_col (str): name of timestamp column
        min_ts (Optional[datetime], optional): ts to filter on. Defaults to None.
        max_ts (Optional[datetime], optional): ts to filter on. Defaults to None.

    Returns:
        pl.DataFrame: _description_
    """
    # filter on min_ts
    if min_ts is not None:
        len_df = len(df)
        df = df.filter(pl.col(ts_col) >= min_ts)
        len_filtered_df = len(df)
        if len_df != len_filtered_df:
            logger.warning(f"Filtered out {len_df - len_filtered_df} rows from {df_name} before min_ts {min_ts}")
    # filter on max_ts
    if max_ts is not None:
        len_df = len(df)
        df = df.filter(pl.col(ts_col) <= max_ts)
        len_filtered_df = len(df)
        if len_df != len_filtered_df:
            logger.warning(f"Filtered out {len_df - len_filtered_df} rows from {df_name} after max_ts {max_ts}")

    return df


#################################################################################################
# Ingest functions for each dataframe
#################################################################################################


def _ingest_app_usage(file_path: Path,
                       min_ts: Optional[datetime] = None,
                       max_ts: Optional[datetime] = None) -> pl.DataFrame:
    
    df = pl.read_csv(file_path, columns=['member_id', 'timestamp']) # event_type is all 'session', don't load it
    
    df = _drop_nulls(df, 'app_usage')  # currently use all columns, the data doesn't contain nulls anyway
    
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    
    # filter timestamp range
    df = _filter_on_timestamp(df, 'app_usage', 'timestamp', min_ts, max_ts)
    
    df = df.sort(by=['member_id', 'timestamp'])
    
    return df


def _ingest_churn_labels(file_path: Path, 
                         min_ts: Optional[datetime] = None,
                         max_ts: Optional[datetime] = None) -> pl.DataFrame:
    df = pl.read_csv(file_path)
    df = _drop_nulls(df, 'churn_labels')  
    
    df = df.with_columns(
        pl.col("signup_date").str.strptime(pl.Datetime, "%Y-%m-%d")
    )
    
    # no timestamp filtering for churn_labels as it only has signup_date
    
    return df


def _ingest_claims(file_path: Path,
                   min_ts: Optional[datetime] = None,
                   max_ts: Optional[datetime] = None) -> pl.DataFrame:
    df = pl.read_csv(file_path)
    df = _drop_nulls(df, 'claims')  # currently use all columns, the data doesn't contain nulls anyway
    
    df = df.with_columns(
        pl.col("diagnosis_date").str.strptime(pl.Datetime, "%Y-%m-%d")
    )
    # filter timestamp range
    df = _filter_on_timestamp(df, 'claims', 'diagnosis_date', min_ts, max_ts)
    
    df.sort(by=['member_id', 'diagnosis_date'])
    return df


def _ingest_web_visits(file_path: Path,
                       min_ts: Optional[datetime] = None,
                       max_ts: Optional[datetime] = None) -> pl.DataFrame:

    relevant_website_titles = ['Healthy eating guide',
                        'Mediterranean diet',
                        'Restorative sleep tips',
                        'Aerobic exercise',
                        'Cardiometabolic health',
                        'Weight management',
                        'Stress reduction',
                        'Sleep hygiene',
                        'HbA1c targets',
                        'Cardio workouts',
                        'High-fiber meals',
                        'Cholesterol friendly foods',
                        'Hypertension basics',
                        'Meditation guide',
                        'Exercise routines',
                        'Diabetes management',
                        'Lowering blood pressure',
                        'Strength training basics',
                        ]
    relevant_columns = ['member_id', 'timestamp', 'title']  # don't use url or description for now
    
    df = pl.read_csv(file_path, columns=relevant_columns)
    
    df = _drop_nulls(df, 'web_visits')  # currently use all columns, the data doesn't contain nulls anyway
    df = df.filter(pl.col("title").is_in(relevant_website_titles))
    
    # convert str to datetime
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    df.sort(by=['member_id', 'timestamp'])
    
    return df

# dict of processing functions for each dataframe, keyed by csv name (e.g. 'app_usage')
ingest_functions = {
    'app_usage': _ingest_app_usage,
    'churn_labels': _ingest_churn_labels,
    'claims': _ingest_claims,
    'web_visits': _ingest_web_visits,
}

def ingest_and_pre_process_data(folder_path: str | Path, 
                                min_ts: Optional[datetime] = None,
                                max_ts: Optional[datetime] = None) -> dict[str, pl.DataFrame]:
    """ Pre-process each df with function based on the csv name.
        Turn str columns to datetime, drop nulls, sort by member_id and timestamp/diagnosis_date
        Select in date range

    Args:
        folder_path: path with all csv files
        min_ts (Optional[datetime], optional): minimum ts to filter on. Defaults to None.
        max_ts (Optional[datetime], optional): maximum ts to filter on. Defaults to None

    Returns:
        dict[str, pl.DataFrame]: preprocessed dict of dataframes, keyed by csv name (e.g. 'app_usage')
    """
    
    folder_path = Path(folder_path)
    dfs = dict()
    
    logger.info(f"Ingesting and pre-processing dataframes from folder: {folder_path}")
    
    for file_path in folder_path.glob('*.csv'):
        file_stem = file_path.stem.replace('test_', '')  # remove 'test_' prefix to unify naming convention
        name = file_stem  # use file_stem as the name key
        
        if name in ingest_functions:
            logger.info(f"Ingesting and pre-processing {name} dataframe")
            dfs[name] = ingest_functions[name](file_path, min_ts, max_ts)
        else:
            logger.warning(f"No ingest function defined for {name}, ingesting without processing")
            dfs[name] = pl.read_csv(file_path)
                
    return dfs