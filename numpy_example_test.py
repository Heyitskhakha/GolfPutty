import pandas as pd

s = pd.Series([10, 20, 30, 40,50])
print(s)
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NY', 'LA', 'SF']
}

df = pd.DataFrame(data)
print(df)