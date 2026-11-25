-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Daily revenue rollup

-- COMMAND ----------

-- DBTITLE 1,Build daily revenue
CREATE OR REPLACE TABLE finance.daily_revenue AS
SELECT order_date, SUM(amount) AS revenue
FROM sales.orders
WHERE status <> 'CANCELLED'
GROUP BY order_date

-- COMMAND ----------

-- MAGIC %python
-- MAGIC threshold = 1000
-- MAGIC display(spark.table("finance.daily_revenue").filter(f"revenue > {threshold}"))

-- COMMAND ----------

MERGE INTO finance.revenue_summary t
USING finance.daily_revenue s
ON t.order_date = s.order_date
WHEN MATCHED THEN UPDATE SET t.revenue = s.revenue
WHEN NOT MATCHED THEN INSERT *
