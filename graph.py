import os
import pandas as pd
import matplotlib.pyplot as plt

iteration = 1
data_choice = 3

filenames = ['results/IAT' + str(iteration) + '.csv', 'results/td' + str(iteration) + '.csv',
             'results/throughput' + str(iteration) + '.csv', 'results/pcount' + str(iteration) + '.csv']

# reading csv file
df = pd.read_csv(filenames[data_choice])

first = df.columns[0]
df.drop([first], axis=1, inplace=True)

subset = df.iloc[0:300]

ax = subset.plot(legend=False, title=filenames[data_choice], lw=1)

ax.set_xlabel("Iteration")
ax.set_ylabel("Value")

plt.show()