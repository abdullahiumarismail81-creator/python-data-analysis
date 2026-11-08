import pandas as pd

data = {
    "Name": ["Ali", "Umar", "Aisha"],
    "Score": [85, 90, 88]
}

df = pd.DataFrame(data)

print(df)
print(df.describe())
