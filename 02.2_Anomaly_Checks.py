from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("RuralCo Anomaly Checks").getOrCreate()

invoice_path = "path/to/invoice.csv"
line_item_path = "path/to/invoice_line_item.csv"

df_invoice = spark.read.option("header", True).option("inferSchema", True).csv(invoice_path)
df_line = spark.read.option("header", True).option("inferSchema", True).csv(line_item_path)
df_invoice.createOrReplaceTempView("invoice")


def has_cols(df, cols):
    """
    Return True if all required columns exist in df.
    Avoids runtime errors when some datasets do not contain certain fields.
    """
    df_cols = set(df.columns)
    return all(c in df_cols for c in cols)



def run_all_checks_spark(df_invoice, df_line):
    """
    Run custom anomaly checks for RuralCo datasets.
    Returns a dictionary: {rule_name: Spark DataFrame}.
    
    Each rule is separated by clear comments for easy modification.
    """
    anomalies = {}

    # Duplicate line_item IDs
    if "id" in df_line.columns:
        duplicated = df_line.groupBy("id").count().filter(F.col("count") > 1)
        anomalies["line_item_duplicate_id"] = df_line.join(
            duplicated.select("id"), on="id", how="inner"
        )

    # line_item.invoice_id not found in invoice.id
    if has_cols(df_line, ["invoice_id"]) and "id" in df_invoice.columns:
        invoice_ids = df_invoice.select("id").distinct().withColumnRenamed("id", "invoice_id_ref")
        anomalies["line_item_missing_invoice_header"] = df_line.join(
            invoice_ids,
            df_line["invoice_id"] == invoice_ids["invoice_id_ref"],
            how="left_anti"
        )

    # Derived gross > received gross
    if has_cols(df_line, ["line_gross_amt_derived", "line_gross_amt_received"]):
        anomalies["derived_gt_received"] = df_line.filter(
            F.col("line_gross_amt_derived") > F.col("line_gross_amt_received")
        )

    # Net > gross (derived amounts)
    if has_cols(df_line, ["line_net_amt_derived", "line_gross_amt_derived"]):
        anomalies["net_exceeds_gross"] = df_line.filter(
            F.col("line_net_amt_derived") > F.col("line_gross_amt_derived")
        )

    # Discount outside normal range
    if "discount_offered" in df_line.columns:
        anomalies["discount_negative"] = df_line.filter(F.col("discount_offered") < 0)
        anomalies["discount_over_100"] = df_line.filter(F.col("discount_offered") > 100)

    # Recalculated gross mismatch
    if has_cols(df_line, ["quantity", "unit_gross", "discount_offered", "line_gross_amt_received"]):
        recalculated = df_line.withColumn(
            "expected_gross",
            F.col("quantity").cast("double")
            * F.col("unit_gross").cast("double")
            * (1 - F.col("discount_offered").cast("double") / 100)
        )
        anomalies["gross_amount_mismatch"] = recalculated.filter(
            F.abs(F.col("line_gross_amt_received").cast("double") - F.col("expected_gross")) > 0.01
        )

    # GST > gross (unusual)
    if has_cols(df_line, ["line_gst_amt_received", "line_gross_amt_received"]):
        anomalies["gst_gt_gross_line"] = df_line.filter(
            F.col("line_gst_amt_received") > F.col("line_gross_amt_received")
        )

    # GST rate exists but gst amount missing
    if has_cols(df_line, ["gst_rate", "line_gst_amt_received"]):
        anomalies["gst_rate_present_gst_missing"] = df_line.filter(
            F.col("gst_rate").isNotNull() & F.col("line_gst_amt_received").isNull()
        )

    # gst_rate = 0.15 but GST amount = 0
    if has_cols(df_line, ["gst_rate", "line_gst_amt_received"]):
        anomalies["gst_rate_15_but_zero_gst"] = df_line.filter(
            (F.col("gst_rate") == 0.15) &
            (F.col("line_gst_amt_received").cast("double") == 0.0)
        )

    # Derived tax consistency
    if has_cols(df_line, ["line_net_amt_derived", "line_gst_total_derived", "line_gross_amt_derived"]):
        tmp = df_line.withColumn(
            "reconstructed_gross",
            F.col("line_net_amt_derived").cast("double") +
            F.col("line_gst_total_derived").cast("double")
        )
        anomalies["derived_tax_inconsistent"] = tmp.filter(
            F.abs(F.col("reconstructed_gross") - F.col("line_gross_amt_derived")) > 0.01
        )

    # created_at > updated_at
    if has_cols(df_line, ["created_at", "updated_at"]):
        tmp_date = df_line.withColumn("_c", F.to_timestamp("created_at")) \
                          .withColumn("_u", F.to_timestamp("updated_at"))
        anomalies["created_after_updated"] = tmp_date.filter(F.col("_c") > F.col("_u"))

    # transaction_date > created_at
    if has_cols(df_line, ["transaction_date", "created_at"]):
        tx = df_line.withColumn("_tx", F.to_date("transaction_date")) \
                    .withColumn("_c", F.to_date(F.to_timestamp("created_at")))
        anomalies["transaction_after_created"] = tx.filter(F.col("_tx") > F.col("_c"))

    # transaction_date < year 2000 (invalid)
    if "transaction_date" in df_line.columns:
        tmp = df_line.withColumn("_tx", F.to_date("transaction_date"))
        anomalies["transaction_date_before_2000"] = tmp.filter(F.col("_tx") < F.lit("2000-01-01"))

    # debit_credit_indicator invalid
    if "debit_credit_indicator" in df_line.columns:
        anomalies["invalid_debit_credit_indicator"] = df_line.filter(
            ~F.col("debit_credit_indicator").isin(["D", "C"])
        )

    # invoice amount mismatch: expected_total = gross_transaction_amount - rebates
    if has_cols(df_invoice, ["total_amount", "gross_transaction_amount", "rebates"]):
        tmp = df_invoice.withColumn(
            "rebates_filled", F.coalesce(F.col("rebates").cast("double"), F.lit(0.0))
        ).withColumn(
            "expected_total",
            F.col("gross_transaction_amount").cast("double") - F.col("rebates_filled")
        )
        anomalies["invoice_amount_mismatch"] = tmp.filter(
            F.abs(F.col("expected_total") - F.col("total_amount")) > 0.01
        )

    return anomalies


def save_anomalies(anomalies, output_dir="./anomaly_results"):
    """
    Save anomalies to separate CSV files.
    
    Args:
        anomalies: Dictionary of {rule_name: DataFrame}
        output_dir: Output directory path
    """
    for rule_name, df_anomaly in anomalies.items():
        if df_anomaly.count() > 0:
            output_path = f"{output_dir}/{rule_name}"
            df_anomaly.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
            print(f"✓ Saved {rule_name}: {df_anomaly.count()} records")
        else:
            print(f"✓ {rule_name}: No anomalies found")


def generate_anomaly_summary(anomalies):
    """
    Generate a summary report of all anomalies.
    
    Args:
        anomalies: Dictionary of {rule_name: DataFrame}
        
    Returns:
        list: Summary with rule names and counts
    """
    summary_data = []
    total_anomalies = 0
    
    for rule_name, df_anomaly in anomalies.items():
        count = df_anomaly.count()
        summary_data.append({
            "rule": rule_name,
            "anomaly_count": count,
            "status": "FAIL" if count > 0 else "PASS"
        })
        total_anomalies += count
    
    print("\n" + "="*70)
    print("ANOMALY CHECK SUMMARY REPORT")
    print("="*70)
    for row in summary_data:
        status_symbol = "❌" if row['status'] == "FAIL" else "✓"
        print(f"{status_symbol} {row['rule']}: {row['anomaly_count']} records")
    print("="*70)
    print(f"Total Anomalies Found: {total_anomalies}")
    print("="*70 + "\n")
    
    return summary_data


def export_summary_to_csv(summary_data, output_path="./anomaly_results/summary_report.csv"):
    """
    Export anomaly summary to CSV file.
    
    Args:
        summary_data: List of summary dictionaries
        output_path: Output file path
    """
    if summary_data:
        spark.createDataFrame(summary_data).write.mode("overwrite").option("header", True).csv(output_path)
        print(f"Summary report saved to {output_path}")


def main():
    """
    Main execution function to run all anomaly checks and generate reports.
    """
    print("Starting RuralCo Anomaly Checks...\n")
    
    # Run anomaly checks
    anomalies = run_all_checks_spark(df_invoice, df_line)
    
    # Generate summary report
    summary = generate_anomaly_summary(anomalies)
    
    # Save anomalies to files
    save_anomalies(anomalies, output_dir="./anomaly_results")
    
    # Export summary
    export_summary_to_csv(summary)
    
    print("Anomaly checks completed!")
    
    return anomalies, summary


if __name__ == "__main__":
    anomalies, summary = main()
