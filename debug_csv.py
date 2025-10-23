import pandas as pd

def debug_csv(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                pd.to_datetime(line.split(',')[0], unit='ms')
            except Exception as e:
                print(f"Error in line {i+1}: {line}")
                print(e)
                break

if __name__ == '__main__':
    debug_csv('BTCUSDT_d.csv')
