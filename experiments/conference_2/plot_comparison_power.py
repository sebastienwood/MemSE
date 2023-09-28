import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('comparison_accuracy.csv')

rmse = np.sqrt(np.mean((df['acc_pred'].to_numpy() - df['acc_ofa'].to_numpy()) ** 2))
print('The RMSE between OFA and Pred is ', rmse)

fig = plt.subplot()
# fig.plot(df['acc_memse'], label='MemSE')
fig.plot(df['acc_pred'], label='PRED')
fig.plot(df['acc_ofa'], label='OFA')
fig.figure.legend()
fig.figure.savefig('comparison_acc.png')

# fig = plt.subplot()
# fig.scatter(df['power_pred'], df['power_memse'])
# fig.set_xlabel('Pred')
# fig.set_ylabel('MemSE')
# fig.figure.savefig('comparison_pow.png')