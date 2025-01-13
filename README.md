import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_date, weekofyear, year, month
import logging

# Step 1: Initialize GlueContext and Logger
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

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

def add_partitions(df: DataFrame) -> DataFrame:
    """Adds Year, Month, and Week partitions to the DataFrame."""
    logger.info("Adding Year, Month, and Week partitions to the DataFrame")
    return df.withColumn("Transaction_Date", to_date(col("TRANSACTION_DATE"), "yyyy-MM-dd")) \
             .withColumn("Year", year(col("Transaction_Date"))) \
             .withColumn("Month", month(col("Transaction_Date"))) \
             .withColumn("Week", weekofyear(col("Transaction_Date")))

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

def write_data_to_s3(df: DataFrame, output_path: str):
    """Writes the DataFrame to S3 with partitions."""
    logger.info(f"Writing data to S3 path: {output_path}")
    df.write.partitionBy("Year", "Month", "Week").mode("overwrite").csv(output_path)

def main():
    try:
        logger.info("Starting the Glue job")

        # Paths
        transactions_path = "s3://your-bucket-name/path_to_transactions_file.csv"
        charges_path = "s3://your-bucket-name/path_to_charges_file.csv"
        output_path = "s3://your-bucket-name/output_path/"

        # Step 2: Read data
        df_transactions = read_data_from_s3(transactions_path)
        df_charges = read_data_from_s3(charges_path)

        # Step 3: Filter and process data
        df_filtered = filter_transactions(df_transactions)
        df_partitioned = add_partitions(df_filtered)
        df_result = calculate_charges(df_partitioned, df_charges)

        # Step 4: Write data to S3
        write_data_to_s3(df_result, output_path)

        logger.info("Glue job completed successfully")

    except Exception as e:
        logger.error(f"Error in Glue job: {str(e)}")
        raise

if __name__ == "__main__":
    main()
