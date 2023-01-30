import argparse

from loguru import logger

from data_preprocessing import load_data, process_data
from modelling import fit, forecast, load_model, save_model
from utils import save_results


def main():
    parser = argparse.ArgumentParser(description="Forecast mid return for cryptoasset")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["fit", "forecast"], 
        help="Mode of operation"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        help="(fit) Path to training data directory with data.h5 and result.h5 or (forecast) path to data.h5"
    )
    args = parser.parse_args()

    mode = args.mode
    data_path = args.data_path

    # Load data
    logger.info("Loading data...")
    OB_df, trades_df = load_data(data_path, mode)
    logger.info("Data was loaded successfully\n")

    # Process data
    logger.info("Processing data...")
    df, TS = process_data(OB_df, trades_df)
    del trades_df
    logger.info("Data processed successfully\n")

    if mode == "fit":
        # Fit model
        logger.info("Training model...")
        model = fit(df)
        logger.info("Model training completed successfully\n")

        # Save model
        logger.info("Saving model...")
        save_model(model)
        logger.info("Model saved to ./models/catboost_model\n")

    elif args.mode == "forecast":

        # Load model
        logger.info("Loading model...")
        model = load_model()
        logger.info("Model loaded successfully\n")
        
        # Forecast with fitted model
        logger.info("Forecasting...")
        y_preds = forecast(df, model)
        logger.info("Forecasting completed successfully\n")

        # Save forecast
        logger.info("Saving results...")
        save_results(y_preds, TS)
        logger.info(f"Results saved in ./forecast.h5\n")


if __name__ == "__main__":
    main()
