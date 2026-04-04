"""
Unit tests for the pipeline
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from src.data_ingestion.load_data import load_data
        from src.data_ingestion.schema_detection import detect_columns
        from src.preprocessing.data_cleaning import clean_data
        from src.feature_engineering.rfm_features import create_rfm
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_schema_detection():
    """Test schema detection"""
    try:
        import pandas as pd
        from src.data_ingestion.schema_detection import detect_columns
        
        # Create mock dataframe
        df = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'InvoiceDate': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'Quantity': [1, 2, 3],
            'UnitPrice': [10.0, 20.0, 30.0]
        })
        
        mapping = detect_columns(df)
        assert mapping['customer_id'] == 'CustomerID'
        print("✅ Schema detection test passed")
        return True
    except Exception as e:
        print(f"❌ Schema detection test failed: {e}")
        return False

if __name__ == "__main__":
    results = [
        test_imports(),
        test_schema_detection()
    ]
    
    if all(results):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)