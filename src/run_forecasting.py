import argparse

from loguru import logger

from utils import save_results
from data_preprocessing import load_data, process_data
from modelling import fit, save_model, load_model, forecast


def main():
    parser = argparse.ArgumentParser(description="Forecast mid return for cryptoasset")
    parser.add_argument("--mode", type=str, choices=["fit", "forecast"], help="Mode of operation")
    parser.add_argument("--train_data_path", type=str, help="Path to training data directory with data.h5 and result.h5")
    parser.add_argument("--forecast_data_path", type=str, help="Path to forecasting data file data.h5")
    parser.add_argument("--output_path", type=str, nargs="?", default="./", help="Directory to save forecast.h5 file (optional)")
    args = parser.parse_args()

    mode = args.mode

    # Load data
    logger.info("Preparing data...")
    OB_df, trades_df = load_data(args.train_path, mode)

    # Process data
    df, TS = process_data(OB_df, trades_df)
    del trades_df

    if mode == "fit":
    
        # Split data
        X = df.drop(columns=["returns"])
        y = df["returns"]
        del df

        # Fit model
        logger.info("Training model...")
        model = fit(X, y)
        logger.info("Model trained successfully\n")

        # Save model
        logger.info("Saving model...")
        save_model(model)
        logger.info("Model saved to ./model\n")

    elif args.mode == "forecast":

        # Load model
        logger.info("Loading model...")
        model = load_model()
        logger.info("Model loaded successfully\n")
        
        # Forecast with fitted model
        logger.info("Forecasting...")
        y_pred = forecast(df, model)
        logger.info(f"Done!\n")

        # Save forecats
        logger.info("Saving results...")
        save_results(y_pred, TS)
        logger.info(f"Results saved in ./forecast.h5\n")

if __name__ == "__main__":
    main()
