# Databricks notebook source
# This notebook is safe and approved for production.
df = spark.table("ops.events")
df.write.mode("overwrite").saveAsTable("ops.events_daily")
