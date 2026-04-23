def stream_data(df, batch_size=5000, delay=0):

    total_rows = len(df)

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size].copy()  # 🔥 FIX
        print(f"📡 Streaming batch: {i} → {i + len(batch)}")

        yield batch