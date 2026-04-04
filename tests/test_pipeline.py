"""
Unit tests for the pipeline
"""
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from src.data_ingestion.load_data import load_data
        from src.data_ingestion.schema_detection import detect_columns
        from src.preprocessing.data_cleaning import clean_data
        from src.feature_engineering.rfm_features import create_rfm
        from src.feature_engineering.multi_source_features import add_multi_source_features
        from src.segmentation.kmeans_segmentation import run_kmeans
        from src.prediction.churn_prediction import train_churn
        from src.prediction.future_prediction import predict_future_purchase
        from src.explainability.shap_explainer import explain_customer
        from src.monitoring.behavior_drift import detect_drift
        from src.model_management.model_versioning import save_models, get_new_version
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_schema_detection():
    """Test schema detection with valid data"""
    try:
        from src.data_ingestion.schema_detection import detect_columns
        
        # Create mock dataframe
        df = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'InvoiceDate': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'Quantity': [1, 2, 3],
            'UnitPrice': [10.0, 20.0, 30.0]
        })
        
        mapping = detect_columns(df)
        assert mapping['customer_id'] == 'CustomerID', "customer_id mapping incorrect"
        assert mapping['transaction_date'] == 'InvoiceDate', "transaction_date mapping incorrect"
        assert mapping['quantity'] == 'Quantity', "quantity mapping incorrect"
        assert mapping['price'] == 'UnitPrice', "price mapping incorrect"
        
        print("✅ Schema detection test passed")
        return True
    except Exception as e:
        print(f"❌ Schema detection test failed: {e}")
        return False

def test_schema_detection_missing_columns():
    """Test schema detection with missing columns"""
    try:
        from src.data_ingestion.schema_detection import detect_columns
        
        # Create mock dataframe with missing columns
        df = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'InvoiceDate': ['2021-01-01', '2021-01-02', '2021-01-03']
        })
        
        try:
            mapping = detect_columns(df)
            print("❌ Should have raised ValueError for missing columns")
            return False
        except ValueError as e:
            if "Missing required columns" in str(e):
                print("✅ Schema detection error handling test passed")
                return True
            else:
                print(f"❌ Wrong error message: {e}")
                return False
    except Exception as e:
        print(f"❌ Schema detection error test failed: {e}")
        return False

def test_data_cleaning():
    """Test data cleaning functionality"""
    try:
        from src.preprocessing.data_cleaning import clean_data
        
        df = pd.DataFrame({
            'CustomerID': [1, 2, 3, 4, None],
            'InvoiceDate': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
            'Quantity': [1, 2, 3, -1, 5],
            'UnitPrice': [10.0, 20.0, 30.0, 40.0, -50.0]
        })
        
        mapping = {
            'customer_id': 'CustomerID',
            'transaction_date': 'InvoiceDate',
            'quantity': 'Quantity',
            'price': 'UnitPrice'
        }
        
        df_clean = clean_data(df, mapping)
        
        assert len(df_clean) == 2, f"Expected 2 rows after cleaning, got {len(df_clean)}"
        assert df_clean['Quantity'].min() > 0, "Found non-positive quantity"
        assert df_clean['UnitPrice'].min() > 0, "Found non-positive price"
        
        print("✅ Data cleaning test passed")
        return True
    except Exception as e:
        print(f"❌ Data cleaning test failed: {e}")
        return False

def test_rfm_creation():
    """Test RFM feature creation"""
    try:
        from src.feature_engineering.rfm_features import create_rfm
        
        df = pd.DataFrame({
            'CustomerID': [1, 1, 2, 2, 3],
            'InvoiceDate': ['2021-01-01', '2021-01-05', '2021-01-02', '2021-01-10', '2021-01-03'],
            'Quantity': [1, 2, 3, 1, 2],
            'UnitPrice': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        mapping = {
            'customer_id': 'CustomerID',
            'transaction_date': 'InvoiceDate',
            'quantity': 'Quantity',
            'price': 'UnitPrice'
        }
        
        rfm = create_rfm(df, mapping)
        
        assert 'Recency' in rfm.columns, "Recency column missing"
        assert 'Frequency' in rfm.columns, "Frequency column missing"
        assert 'Monetary' in rfm.columns, "Monetary column missing"
        assert len(rfm) == 3, f"Expected 3 customers, got {len(rfm)}"
        
        print("✅ RFM creation test passed")
        return True
    except Exception as e:
        print(f"❌ RFM creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 RUNNING PIPELINE TESTS")
    print("="*50 + "\n")
    
    results = [
        test_imports(),
        test_schema_detection(),
        test_schema_detection_missing_columns(),
        test_data_cleaning(),
        test_rfm_creation()
    ]
    
    print("\n" + "="*50)
    if all(results):
        print(f"✅ ALL TESTS PASSED ({len(results)}/{len(results)})")
        print("="*50 + "\n")
        sys.exit(0)
    else:
        passed = sum(results)
        print(f"❌ SOME TESTS FAILED ({passed}/{len(results)} passed)")
        print("="*50 + "\n")
        sys.exit(1)