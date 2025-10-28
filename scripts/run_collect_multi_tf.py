from quantylab.rltrader import data_manager

print('Starting multi-tf collection (7 days)...')
try:
    df = data_manager.load_multi_tf_crypto_data(ticker='KRW-BTC', days=7, intervals=['minute1','minute5','minute15','minute60','minute240','day'], base_interval='minute1')
    print('Collected shape:', None if df is None else df.shape)
    if df is not None:
        print('Columns sample:', df.columns[:20].tolist())
        import os
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/KRW-BTC_multi_tf_sample.csv', index=False)
        print('Wrote data/KRW-BTC_multi_tf_sample.csv')
except Exception as e:
    print('Error during collection:', e)
