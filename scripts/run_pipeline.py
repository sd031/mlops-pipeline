"""
Complete MLOps pipeline runner script.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, MLOpsLogger
from src.utils.helpers import load_config, ensure_dir
from src.utils.database import db_manager
from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator

logger = get_logger(__name__)
mlops_logger = MLOpsLogger(__name__)


class MLOpsPipeline:
    """Complete MLOps pipeline orchestrator."""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize MLOps pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.pipeline_id = f"pipeline_{int(time.time())}"
        
        # Initialize components
        self.data_ingestion = DataIngestion(self.config)
        self.data_validator = DataValidator(self.config)
        
        logger.info(f"MLOps Pipeline initialized: {self.pipeline_id}")
    
    def run_data_pipeline(self):
        """
        Run the data pipeline (ingestion and validation).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting data pipeline...")
            
            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            raw_data_path = "data/raw/customer_churn.csv"
            
            if not os.path.exists(raw_data_path):
                logger.error(f"Raw data file not found: {raw_data_path}")
                logger.info("Please run 'python scripts/generate_data.py' first to generate sample data")
                return False
            
            # Ingest data
            df = self.data_ingestion.ingest_csv(raw_data_path, "ingested_data")
            logger.info(f"Data ingested: {len(df)} rows, {len(df.columns)} columns")
            
            # Step 2: Data Validation
            logger.info("Step 2: Data Validation")
            validation_results = self.data_validator.run_full_validation(df, "churn")
            
            if not validation_results['validation_passed']:
                logger.error("Data validation failed!")
                for error in validation_results['errors']:
                    logger.error(f"  - {error}")
                return False
            
            logger.info("Data validation passed!")
            
            # Step 3: Save processed data
            logger.info("Step 3: Saving processed data")
            ensure_dir("data/processed")
            processed_path = "data/processed/validated_data.csv"
            df.to_csv(processed_path, index=False)
            logger.info(f"Processed data saved to: {processed_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            return False
    
    def run_training_pipeline(self):
        """
        Run the model training pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting training pipeline...")
            
            # Import training modules
            from src.models.train import ModelTrainer
            
            # Initialize trainer
            trainer = ModelTrainer(self.config)
            
            # Load processed data
            processed_data_path = "data/processed/validated_data.csv"
            if not os.path.exists(processed_data_path):
                logger.error(f"Processed data not found: {processed_data_path}")
                return False
            
            # Train model
            model_info = trainer.train_model(processed_data_path)
            
            if model_info:
                logger.info(f"Model training completed: {model_info['model_path']}")
                return True
            else:
                logger.error("Model training failed")
                return False
                
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False
    
    def run_evaluation_pipeline(self):
        """
        Run the model evaluation pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting evaluation pipeline...")
            
            # Import evaluation modules
            from src.models.evaluate import ModelEvaluator
            
            # Initialize evaluator
            evaluator = ModelEvaluator(self.config)
            
            # Run evaluation
            evaluation_results = evaluator.evaluate_latest_model()
            
            if evaluation_results:
                logger.info("Model evaluation completed successfully")
                for metric, value in evaluation_results['metrics'].items():
                    logger.info(f"  {metric}: {value:.4f}")
                return True
            else:
                logger.error("Model evaluation failed")
                return False
                
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}")
            return False
    
    def run_deployment_pipeline(self):
        """
        Run the model deployment pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting deployment pipeline...")
            
            # For this demo, we'll just verify the model is ready for deployment
            models_dir = "models/artifacts"
            if not os.path.exists(models_dir):
                logger.error("No trained models found for deployment")
                return False
            
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if not model_files:
                logger.error("No model artifacts found for deployment")
                return False
            
            latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
            logger.info(f"Latest model ready for deployment: {latest_model}")
            
            # In a real deployment, you would:
            # 1. Run model validation tests
            # 2. Deploy to staging environment
            # 3. Run integration tests
            # 4. Deploy to production
            # 5. Update model registry
            
            logger.info("Model deployment pipeline completed (demo mode)")
            return True
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")
            return False
    
    def run_full_pipeline(self):
        """
        Run the complete MLOps pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        logger.info(f"Starting full MLOps pipeline: {self.pipeline_id}")
        
        try:
            # Initialize database
            db_manager.init_db()
            
            # Step 1: Data Pipeline
            if not self.run_data_pipeline():
                logger.error("Data pipeline failed, stopping execution")
                return False
            
            # Step 2: Training Pipeline
            if not self.run_training_pipeline():
                logger.error("Training pipeline failed, stopping execution")
                return False
            
            # Step 3: Evaluation Pipeline
            if not self.run_evaluation_pipeline():
                logger.error("Evaluation pipeline failed, stopping execution")
                return False
            
            # Step 4: Deployment Pipeline
            if not self.run_deployment_pipeline():
                logger.error("Deployment pipeline failed, stopping execution")
                return False
            
            # Pipeline completed successfully
            total_time = time.time() - start_time
            logger.info(f"Full MLOps pipeline completed successfully in {total_time:.2f} seconds")
            
            # Log pipeline completion
            mlops_logger.log_experiment(
                experiment_name="full_pipeline",
                run_id=self.pipeline_id,
                metrics={"pipeline_duration": total_time, "success": 1}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Full pipeline failed: {e}")
            return False


def main():
    """Main function for pipeline execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLOps Pipeline Runner")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--pipeline", default="full", 
                       choices=["data", "training", "evaluation", "deployment", "full"],
                       help="Pipeline to run")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLOpsPipeline(args.config)
    
    # Run specified pipeline
    success = False
    if args.pipeline == "data":
        success = pipeline.run_data_pipeline()
    elif args.pipeline == "training":
        success = pipeline.run_training_pipeline()
    elif args.pipeline == "evaluation":
        success = pipeline.run_evaluation_pipeline()
    elif args.pipeline == "deployment":
        success = pipeline.run_deployment_pipeline()
    elif args.pipeline == "full":
        success = pipeline.run_full_pipeline()
    
    # Exit with appropriate code
    if success:
        print("✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
