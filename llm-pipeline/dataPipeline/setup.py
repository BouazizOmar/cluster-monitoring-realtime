# setup.py
from setuptools import setup, find_packages

setup(
  name="data_pipeline",
  version="0.1",
  packages=find_packages(),  # will pick up data_ingestion, metrics_processing, etc.
)
