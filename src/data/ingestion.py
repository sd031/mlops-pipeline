"""
Data ingestion module for the MLOps pipeline.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests
from datetime import datetime
import json

from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, get_timestamp
from src.utils.database import db_manager

logger = get_logger(__name__)


class DataIngestion:
    """Data ingestion class for various data sources."""
    
    def __init__(self, config: Dict):
        """
        Initialize data ingestion.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.raw_data_path = config.get('data', {}).get('raw_path', 'data/raw')
        ensure_dir(self.raw_data_path)
    
    def ingest_csv(self, file_path: str, output_name: Optional[str] = None) -> pd.DataFrame:
        """
        Ingest data from CSV file.
        
        Args:
            file_path: Path to CSV file
            output_name: Output file name (optional)
            
        Returns:
            Ingested DataFrame
        """
        try:
            logger.info(f"Ingesting CSV data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Save to raw data directory if output_name provided
            if output_name:
                output_path = os.path.join(self.raw_data_path, f"{output_name}_{get_timestamp()}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved ingested data to {output_path}")
            
            # Log data quality metrics
            self._log_ingestion_metrics(df, file_path)
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting CSV data: {e}")
            raise
    
    def ingest_json(self, file_path: str, output_name: Optional[str] = None) -> pd.DataFrame:
        """
        Ingest data from JSON file.
        
        Args:
            file_path: Path to JSON file
            output_name: Output file name (optional)
            
        Returns:
            Ingested DataFrame
        """
        try:
            logger.info(f"Ingesting JSON data from {file_path}")
            df = pd.read_json(file_path)
            
            if output_name:
                output_path = os.path.join(self.raw_data_path, f"{output_name}_{get_timestamp()}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved ingested data to {output_path}")
            
            self._log_ingestion_metrics(df, file_path)
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting JSON data: {e}")
            raise
    
    def ingest_from_api(self, url: str, headers: Optional[Dict] = None, 
                       params: Optional[Dict] = None, output_name: Optional[str] = None) -> pd.DataFrame:
        """
        Ingest data from API endpoint.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            params: Request parameters
            output_name: Output file name (optional)
            
        Returns:
            Ingested DataFrame
        """
        try:
            logger.info(f"Ingesting data from API: {url}")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            if output_name:
                output_path = os.path.join(self.raw_data_path, f"{output_name}_{get_timestamp()}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved ingested data to {output_path}")
            
            self._log_ingestion_metrics(df, url)
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting data from API: {e}")
            raise
    
    def ingest_database(self, connection_string: str, query: str, 
                       output_name: Optional[str] = None) -> pd.DataFrame:
        """
        Ingest data from database.
        
        Args:
            connection_string: Database connection string
            query: SQL query
            output_name: Output file name (optional)
            
        Returns:
            Ingested DataFrame
        """
        try:
            logger.info(f"Ingesting data from database")
            df = pd.read_sql(query, connection_string)
            
            if output_name:
                output_path = os.path.join(self.raw_data_path, f"{output_name}_{get_timestamp()}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved ingested data to {output_path}")
            
            self._log_ingestion_metrics(df, "database")
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting data from database: {e}")
            raise
    
    def batch_ingest(self, sources: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Batch ingest data from multiple sources.
        
        Args:
            sources: List of source configurations
            
        Returns:
            Dictionary of DataFrames
        """
        results = {}
        
        for source in sources:
            source_type = source.get('type')
            source_name = source.get('name')
            
            try:
                if source_type == 'csv':
                    df = self.ingest_csv(source['path'], source_name)
                elif source_type == 'json':
                    df = self.ingest_json(source['path'], source_name)
                elif source_type == 'api':
                    df = self.ingest_from_api(
                        source['url'], 
                        source.get('headers'), 
                        source.get('params'),
                        source_name
                    )
                elif source_type == 'database':
                    df = self.ingest_database(
                        source['connection_string'],
                        source['query'],
                        source_name
                    )
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    continue
                
                results[source_name] = df
                logger.info(f"Successfully ingested data from {source_name}")
                
            except Exception as e:
                logger.error(f"Failed to ingest data from {source_name}: {e}")
                continue
        
        return results
    
    def _log_ingestion_metrics(self, df: pd.DataFrame, source: str):
        """
        Log data ingestion metrics.
        
        Args:
            df: Ingested DataFrame
            source: Data source
        """
        metrics = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Log to database
        db_manager.log_data_quality(
            dataset_name=f"ingestion_{source}",
            metrics=metrics,
            data_version=get_timestamp()
        )
        
        logger.info(f"Ingestion metrics for {source}: {metrics}")


def main():
    """Main function for data ingestion CLI."""
    import argparse
    from src.utils.helpers import load_config
    
    parser = argparse.ArgumentParser(description="Data Ingestion")
    parser.add_argument("--source", required=True, help="Data source path")
    parser.add_argument("--type", default="csv", choices=["csv", "json", "api", "database"],
                       help="Data source type")
    parser.add_argument("--output", help="Output name")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize data ingestion
    ingestion = DataIngestion(config)
    
    # Ingest data based on type
    if args.type == "csv":
        df = ingestion.ingest_csv(args.source, args.output)
    elif args.type == "json":
        df = ingestion.ingest_json(args.source, args.output)
    else:
        logger.error(f"CLI ingestion not implemented for type: {args.type}")
        return
    
    logger.info(f"Successfully ingested {len(df)} rows and {len(df.columns)} columns")


if __name__ == "__main__":
    main()
