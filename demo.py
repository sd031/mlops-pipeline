#!/usr/bin/env python3
"""
MLOps Pipeline Demo Script

This script demonstrates the complete MLOps pipeline functionality.
Run this to see the end-to-end workflow in action.
"""
import os
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

def print_banner(text):
    """Print a banner with the given text."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step_num, description):
    """Print a step description."""
    print(f"\n🔄 Step {step_num}: {description}")
    print("-" * 50)

def run_demo():
    """Run the complete MLOps pipeline demo."""
    print_banner("MLOps Pipeline Demo")
    print("Welcome to the comprehensive MLOps pipeline demonstration!")
    print("This demo will showcase:")
    print("• Data generation and ingestion")
    print("• Data validation and quality checks")
    print("• Model training with experiment tracking")
    print("• Model evaluation and metrics")
    print("• API deployment and serving")
    print("• Monitoring and drift detection")
    
    try:
        # Step 1: Generate Sample Data
        print_step(1, "Generating Sample Customer Churn Data")
        os.system("python scripts/generate_data.py")
        print("✅ Sample data generated successfully!")
        
        # Step 2: Run Data Pipeline
        print_step(2, "Running Data Pipeline (Ingestion & Validation)")
        result = os.system("python scripts/run_pipeline.py --pipeline data")
        if result == 0:
            print("✅ Data pipeline completed successfully!")
        else:
            print("❌ Data pipeline failed!")
            return False
        
        # Step 3: Train Model
        print_step(3, "Training Machine Learning Model")
        result = os.system("python scripts/run_pipeline.py --pipeline training")
        if result == 0:
            print("✅ Model training completed successfully!")
        else:
            print("❌ Model training failed!")
            return False
        
        # Step 4: Evaluate Model
        print_step(4, "Evaluating Model Performance")
        result = os.system("python scripts/run_pipeline.py --pipeline evaluation")
        if result == 0:
            print("✅ Model evaluation completed successfully!")
        else:
            print("❌ Model evaluation failed!")
            return False
        
        # Step 5: Show Project Structure
        print_step(5, "Project Structure Overview")
        print("📁 Generated project structure:")
        os.system("tree -L 3 -I '__pycache__|*.pyc|.git' || find . -type d -not -path '*/.*' | head -20")
        
        # Step 6: Show Available Commands
        print_step(6, "Available Commands")
        print("🛠️  You can now use these commands:")
        print("   make generate-data     # Generate new sample data")
        print("   make train-model       # Train a new model")
        print("   make serve-model       # Start the API server")
        print("   make docker-up         # Start all services with Docker")
        print("   make test              # Run the test suite")
        print("   make run-pipeline      # Run the complete pipeline")
        
        print_step(7, "Next Steps")
        print("🚀 To start the API server:")
        print("   python -m src.api.app")
        print("   # Then visit http://localhost:8000/docs for API documentation")
        print("\n🐳 To start all services with Docker:")
        print("   docker-compose up -d")
        print("   # This will start MLflow, PostgreSQL, API, and monitoring services")
        
        print_banner("Demo Completed Successfully! 🎉")
        print("Your MLOps pipeline is ready for production use!")
        print("Check the README.md for detailed documentation and usage instructions.")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        return False

if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
