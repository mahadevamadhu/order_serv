import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import logging
import datetime
import os

# Step 1: Initialize GlueContext and Logger
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_path(base_path: str, cycle_number: str = "0", date_format: str = "%d%m%Y%H%M") -> str:
    """Generates the file path based on the cycle number and current date-time."""
    current_time = datetime.datetime.now().strftime(date_format)
    file_name = f"SETELEMENT_JOBS_{cycle_number}_{current_time}.xlsx"
    file_path = os.path.join(base_path, file_name)
    logger.info(f"Generated file path: {file_path}")
    return file_path

def read_data_from_s3(s3_path: str, format: str = "csv", has_header: bool = True) -> DataFrame:
    """Reads data from an S3 location and returns a DataFrame."""
    logger.info(f"Reading data from S3 path: {s3_path}")
    return glueContext.create_dynamic_frame.from_options(
        connection_type="s3",
        connection_options={"paths": [s3_path]},
        format=format,
        format_options={"withHeader": has_header}
    ).toDF()

def filter_transactions(df: DataFrame) -> DataFrame:
    """Filters transactions based on RESPONSE_CODE and TRANSACTION_STATUS."""
    logger.info("Filtering transactions with RESPONSE_CODE=0 and TRANSACTION_STATUS=TRANSACTION_APPROVED")
    return df.filter((col("RESPONSE_CODE") == "0") & (col("TRANSACTION_STATUS") == "TRANSACTION_APPROVED"))

def calculate_charges(df: DataFrame, charges_df: DataFrame) -> DataFrame:
    """Joins the transactions DataFrame with charges and calculates charges and net amount."""
    logger.info("Joining transactions with charges and calculating charges and net amount")
    df = df.join(charges_df, ["MERCHANTID", "MCC_CODE"], "left")
    df = df.withColumn("MDR_Rate", col("MDR_Rate").cast("double")) \
           .withColumn("Instrument_Rate", col("Instrument_Rate").cast("double")) \
           .withColumn("GST_Rate", col("GST_Rate").cast("double"))

    return df.withColumn("MDR_Charges", col("AMOUNT") * col("MDR_Rate")) \
             .withColumn("Instrument_Charges", col("AMOUNT") * col("Instrument_Rate")) \
             .withColumn("GST", (col("MDR_Charges") + col("Instrument_Charges")) * col("GST_Rate")) \
             .withColumn("Net_Amount", col("AMOUNT") - (col("MDR_Charges") + col("Instrument_Charges") + col("GST")))

def write_data_to_mongodb(df: DataFrame, uri: str, database: str, collection: str):
    """Writes the DataFrame to MongoDB."""
    logger.info(f"Writing data to MongoDB database: {database}, collection: {collection}")
    df.write.format("mongo") \
        .option("uri", uri) \
        .option("database", database) \
        .option("collection", collection) \
        .mode("overwrite") \
        .save()

def main():
    try:
        logger.info("Starting the Glue job")

        # Paths and MongoDB connection details
        base_path = "s3://your-bucket-name/path_to_files/"
        cycle_number = "0"  # This should be dynamically set if needed
        transactions_path = get_file_path(base_path, cycle_number)
        charges_path = "s3://your-bucket-name/path_to_charges_file.csv"
        mongo_uri = "mongodb://your_mongo_host:your_mongo_port"
        mongo_database = "your_database_name"
        mongo_collection = "filtered_transactions"

        # Step 2: Read data
        df_transactions = read_data_from_s3(transactions_path)
        df_charges = read_data_from_s3(charges_path)

        # Step 3: Filter and process data
        df_filtered = filter_transactions(df_transactions)
        df_result = calculate_charges(df_filtered, df_charges)

        # Step 4: Write data to MongoDB
        write_data_to_mongodb(df_result, mongo_uri, mongo_database, mongo_collection)

        logger.info("Glue job completed successfully")

    except Exception as e:
        logger.error(f"Error in Glue job: {str(e)}")
        raise

if __name__ == "__main__":
    main()
