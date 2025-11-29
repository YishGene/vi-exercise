from datetime import datetime
import polars as pl
import logging
# obs_window_end = datetime(2025, 7, 16)  # start of day after end of observation window

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _extract_claims_features(claims_df: pl.DataFrame, obs_window_end: datetime) -> pl.DataFrame:
    
    # group claims by member_id and icd_code to get first and last diagnosis dates and number of diagnoses
    claims_per_id = claims_df.group_by(
    ['member_id', 'icd_code']).agg(
        pl.min('diagnosis_date').alias('first_dx_date'),
        pl.max('diagnosis_date').alias('last_dx_date'),
        pl.count('diagnosis_date').alias('dx_count')
    )

    # diff from obs_window_end, in days
    claims_per_id = claims_per_id.with_columns(
        (obs_window_end - pl.col("first_dx_date")).dt.total_days().alias("first_dx_dt"),
        (obs_window_end - pl.col("last_dx_date")).dt.total_days().alias("last_dx_dt"),
    )
    # pivot to use as columns    
    claims_per_id = claims_per_id.pivot(values=['dx_count', 'first_dx_dt', 'last_dx_dt'],
                                        index='member_id',
                                        on='icd_code').fill_null(0)
    
    return claims_per_id


def _extract_claims_features_no_single(claims_df: pl.DataFrame, obs_window_end: datetime) -> pl.DataFrame:
    
    # group claims by member_id and icd_code to get first and last diagnosis dates and number of diagnoses
    claims_per_id = claims_df.group_by(
    ['member_id']).agg(
        pl.min('diagnosis_date').alias('first_dx_date'),
        pl.max('diagnosis_date').alias('last_dx_date'),
        pl.count('diagnosis_date').alias('dx_count')
    )

    # diff from obs_window_end, in days
    claims_per_id = claims_per_id.with_columns(
        (obs_window_end - pl.col("first_dx_date")).dt.total_days().alias("first_dx_dt"),
        (obs_window_end - pl.col("last_dx_date")).dt.total_days().alias("last_dx_dt"),
    )

    claims_per_id = claims_per_id[['member_id', 'dx_count', 'first_dx_dt', 'last_dx_dt']]
    
    return claims_per_id



def _extract_web_visits_features(app_usage_df: pl.DataFrame, obs_window_end: datetime) -> pl.DataFrame:
    # group only by member id to get first and last web_visit dates and number of web_visits, this treats all website visits the same
    
    # TODO: move filtering by title column to here from ingestion?
    web_visits_per_id = app_usage_df.group_by(
        ['member_id']).agg(
            pl.min('timestamp').alias('first_wv_date'),
            pl.max('timestamp').alias('last_wv_date'),
            pl.count('timestamp').alias('wv_count')
    )

    # diff from obs_window_end, in days
    web_visits_per_id = web_visits_per_id.with_columns(
        (obs_window_end - pl.col("first_wv_date")).dt.total_days().alias("first_wv_dt"),
        (obs_window_end - pl.col("last_wv_date")).dt.total_days().alias("last_wv_dt"),
    )

    web_visits_per_id = web_visits_per_id[['member_id', 'wv_count', 'first_wv_dt', 'last_wv_dt']]
    
    return web_visits_per_id


def _extract_app_usage_features(app_usage_df: pl.DataFrame, obs_window_end: datetime) -> pl.DataFrame:
    app_usage_per_id = app_usage_df.group_by(
    ['member_id']).agg(
        pl.min('timestamp').alias('first_au_date'),
        pl.max('timestamp').alias('last_au_date'),
        pl.count('timestamp').alias('au_count')
    )

    # diff from obs_window_end, in days
    app_usage_per_id = app_usage_per_id.with_columns(
        (obs_window_end - pl.col("first_au_date")).dt.total_days().alias("first_au_dt"),
        (obs_window_end - pl.col("last_au_date")).dt.total_days().alias("last_au_dt"),
    )

    app_usage_per_id = app_usage_per_id[['member_id', 'au_count', 'first_au_dt', 'last_au_dt']]
    
    return app_usage_per_id
    

def featurize_data(dfs: dict[str, pl.DataFrame], 
                   obs_window_end: datetime = datetime(2025, 7, 16), 
                   fill_nulls: bool = True) -> pl.DataFrame:
    """ Takes all loaded df-s and outputs a single df with all engineered features. 

    Args:
        dfs (dict[str, pl.DataFrame]): _description_
        obs_window_end (datetime, optional): start of day after end of observation window. Defaults to datetime(2025, 7, 16).
        fill_nulls (bool, optional): whether to fill nulls in the final feature set. Defaults to True.

    Returns:
        pl.DataFrame: _description_
    """
    logger.info("Starting featurization...")   
    features_w_labels = dfs['churn_labels'].clone()
    

    features_w_labels = features_w_labels.with_columns(
        (obs_window_end - pl.col("signup_date")).dt.total_days().alias("signup_date_dt")
    )
    features_w_labels = features_w_labels.drop('signup_date')
    
    # Feature set 1: claims features 
    # ------------------------------
    # claims_per_id = _extract_claims_features(dfs['claims'],
    #                                          obs_window_end)
    logger.info("Extracting claims features...")
    claims_per_id = _extract_claims_features_no_single(dfs['claims'],
                                                       obs_window_end)
    
    # merge to features matrix
    features_w_labels = features_w_labels.join(claims_per_id, on='member_id', how='left')
    # fill nulls: counts filled with 0, non-existant dates filled with large number (days from obs_window_end)
    # features_w_labels = features_w_labels.with_columns(
    #     pl.col(r"^dx_count_.*$").fill_null(0),
    #     pl.col(r"^first_dx_dt.*$").fill_null(1e5),
    #     pl.col(r"^last_dx_dt.*$").fill_null(1e5),
    # )
    if fill_nulls:
        features_w_labels = features_w_labels.with_columns(
            pl.col("dx_count").fill_null(0),
            pl.col("first_dx_dt").fill_null(1e5),
            pl.col("last_dx_dt").fill_null(1e5),
        )
    
    # Feature set 2: web_visits features
    # ---------------------------------
    logger.info("Extracting web_visits features...")
    web_visits_per_id = _extract_web_visits_features(dfs['web_visits'],
                                                     obs_window_end)
    # merge to features matrix
    features_w_labels = features_w_labels.join(web_visits_per_id, on='member_id', how='left')
    # fill nulls: counts filled with 0, non-existant dates filled with large number (days from obs_window_end)
    if fill_nulls:
        features_w_labels = features_w_labels.with_columns(
            pl.col("wv_count").fill_null(0),
            pl.col("first_wv_dt").fill_null(1e5),
            pl.col("last_wv_dt").fill_null(1e5),
        )
    
    # Feature set 3: app_usage features
    # ---------------------------------
    logger.info("Extracting app_usage features...")
    app_usage_per_id = _extract_app_usage_features(dfs['app_usage'],
                                                   obs_window_end)
    # merge to features matrix
    features_w_labels = features_w_labels.join(app_usage_per_id, on='member_id', how='left')
    # fill nulls: counts filled with 0, non-existant dates filled with large number (days from obs_window_end)
    if fill_nulls:
        features_w_labels = features_w_labels.with_columns(
            pl.col("au_count").fill_null(0),
            pl.col("first_au_dt").fill_null(1e5),
            pl.col("last_au_dt").fill_null(1e5),
        )
    
    return features_w_labels