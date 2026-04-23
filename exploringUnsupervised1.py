import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt

kagglehub.login()
# Download latest version
# path = kagglehub.competition_download('titanic')
# new_path = os.path.join(path + "/Users/mix/.cache/kagglehub/competitions/titanic/train.csv")
new_path = os.path.join("/Users/mix/.cache/kagglehub/competitions/titanic/train.csv")
print("Path to dataset files:", new_path)

data = pd.read_csv(new_path)


