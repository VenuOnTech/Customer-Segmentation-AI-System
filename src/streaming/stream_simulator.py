import pandas as pd
import time

def stream_data(df, batch_size=5000, delay=1):
    """
    Simulate streaming data in batches
    """
    total_rows = len(df)

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size]
        print(f"📡 Streaming batch: {i} → {i + len(batch)}")

        yield batch
        time.sleep(delay)