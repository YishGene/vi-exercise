
import polars as pl
from sklift.metrics import uplift_at_k, qini_auc_score


from econml.metalearners import XLearner
from lightgbm import LGBMClassifier

import logging
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


## CATE model training

def train_cate(features_w_labels: pl.DataFrame):
    logger.info("Training CATE model...")
    # drop the labels
    features = features_w_labels.drop(['member_id', 'churn', 'outreach'])

    # T = outreach, Y = churn
    cate_model = XLearner(models=LGBMClassifier(max_depth=5, 
                                            n_estimators=1000,
                                            reg_alpha=0.1,
                                            reg_lambda=0.1,
                                            verbosity=-1))
    # note that fitting is for 1-chrun, since we want to model P(no_churn)
    # Treatment is outreach
    cate_model.fit(Y=1-features_w_labels['churn'], 
                   T=features_w_labels['outreach'], 
                   X=features)

    return cate_model


def cate_inference(cate_model, features_w_labels: pl.DataFrame) -> np.ndarray:
    logger.info("Inference with CATE model...")
    features = features_w_labels.drop(['member_id', 'churn', 'outreach'])
    # Get conditional treatment effect
    te = cate_model.effect(features)   # E[Y|T=1,X] - E[Y|T=0,X]
    return te


def evaluate_cate(cate_model, features_w_labels: pl.DataFrame) -> pl.DataFrame:
    """_summary_

    Args:
        cate_model (_type_): _description_
        features_w_labels (pl.DataFrame): _description_

    Returns:
        pl.DataFrame: eval dataframe with extra column: te (treatment effect) and labels outreach, churn
    """
    logger.info("Evaluating CATE model...")
    
    te = cate_inference(cate_model, features_w_labels)
    
    eval_df = pl.DataFrame({'te': te, 
                        'member_id': features_w_labels['member_id'],
                        'outreach': features_w_labels['outreach'],
                        'churn': features_w_labels['churn'],})

    return eval_df