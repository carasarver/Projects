import pandas as pd
import os

os.chdir("C:\\Cara\\Documents\\Coding\\Datasets\\Climate Data")

climatedata1 = pd.read_csv("Climate Data_2019.01001099999.csv")

print(climatedata1.head(10))