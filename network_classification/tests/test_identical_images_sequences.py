import pandas as pd

df1 = pd.read_csv("rotation_sequence_A.csv")
df2 = pd.read_csv("rotation_sequence_B.csv")

column_name = "filename"

names1 = df1[column_name].astype(str)
names2 = df2[column_name].astype(str)

common_names = set(names1) & set(names2)

if common_names:
    print("identical names:")
    for name in common_names:
        print(name)
else:
    print("no identical names.")
