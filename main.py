from argparse import ArgumentParser
from data_ingestion import ingest_and_pre_process_data
from featurization import featurize_data
from model import train_cate, evaluate_cate
from pathlib import Path
import polars as pl

from sklift.metrics import uplift_at_k, qini_auc_score
from sklift.viz import plot_qini_curve

import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = ArgumentParser(description="Run the data ingestion and featurization pipeline.")
    parser.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Path to the folder containing test and train folders, each with all the CSV files."
    )
    parser.add_argument(
        "--output-folder", 
        type=str,
        default="output",
        help="Path to the folder to save outputs."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    data_folder = Path(args.data_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    ######## Train ########
    # Ingest and pre-process train data
    print('should log now')
    logger.info(f"Processing train data from {data_folder / 'train'}")
    train_dfs = ingest_and_pre_process_data(data_folder / 'train')
    # Featurize data
    train_features_w_label = featurize_data(train_dfs)
    # Train CATE model
    cate_model = train_cate(train_features_w_label)

    logger.info("Evaluating train data...")
    train_eval_df = evaluate_cate(cate_model, train_features_w_label)
    logger.info("Writing train report...")
    write_report(str(output_folder / 'train'), train_eval_df)
    
    
    ######## Test ########
    # Ingest and pre-process test data
    logger.info(f"Processing test data from {data_folder / 'test'}")
    test_dfs = ingest_and_pre_process_data(data_folder / 'test')
    # Featurize data
    test_features_w_label = featurize_data(test_dfs)
    logger.info("Evaluating test data...")
    test_eval_df = evaluate_cate(cate_model, test_features_w_label)
    logger.info("Writing test report...")
    write_report(str(output_folder / 'test'), test_eval_df)

def write_report(out_dir: str, eval_df: pl.DataFrame):
    """ writes the csv, the confusion matrix png, and the text report with metrics to the out_dir

    Args:
        out_dir (str): 
        results_df (pd.DataFrame): must contain columns true_class, pred_class
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # results csv
    # ----------------------------------
    top_n_csv_path = out_path / '.csv'
    logger.info(f"Results saved to {top_n_csv_path}")
    
    results_top_n = eval_df.clone()

    results_top_n = results_top_n.rename({'te': 'prioritization_score'})
    results_top_n = results_top_n.sort(by='prioritization_score', descending=True)
    results_top_n = results_top_n.with_columns(pl.int_range(1, results_top_n.height + 1).alias('rank'))

    results_top_n = results_top_n[['member_id', 'prioritization_score', 'rank']]
    results_top_n.write_csv(top_n_csv_path)
    
    # qini curve
    # ----------------------------------
    plt.figure()
    plot_qini_curve(y_true=1-eval_df['churn'],
                uplift=eval_df['te'],
                treatment=eval_df['outreach'],
                )
    plt.legend(loc='upper right')
    plt.savefig(out_path / 'qini_curve.png')
    plt.close()
    logger.info(f"Qini curve saved to {out_path / 'qini_curve.png'}")
    
    # metrics:
    # ----------------------------------
    auuc_score = qini_auc_score(y_true=1-eval_df['churn'], 
                                uplift=eval_df['te'], 
                                treatment=eval_df['outreach'])
    logger.info(f"Train AUUC Score: {auuc_score}")


    report_txt_path = out_path / 'report.txt'
    with open(report_txt_path, 'w') as f:
        f.write(f"AUUC Score: {auuc_score:.3f}\n")
        
    logger.info(f"Report saved to {report_txt_path}")

if __name__ == "__main__":
    main()
