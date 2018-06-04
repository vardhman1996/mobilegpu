import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set()
df = pd.read_csv('../exp_mat_mul_3000_batch.csv', sep='|')

set_batches = set([(shapeA, shapeB) for shapeA, shapeB in zip(df['shapeA'], df['shapeB'])])

for batch in set_batches:
    plt.figure()
    df_filt = df[df['shapeA'] == batch[0]]
    df_filt = df_filt[df_filt['shapeB'] == batch[1]]

    line_xf = df_filt[['x_f', 'time_taken']].groupby('x_f').min().reset_index().values
    line_yf = df_filt[['y_f', 'time_taken']].groupby('y_f').min().reset_index().values
    line_kf = df_filt[['k_f', 'time_taken']].groupby('k_f').min().reset_index().values

    plt.plot(line_xf[:, 0], line_xf[:, 1], label='row tiling factor')
    plt.plot(line_yf[:, 0], line_yf[:, 1], label='column tiling factor')
    plt.plot(line_kf[:, 0], line_kf[:, 1], label='reduce axis tiling factor')

    plt.xlabel('value of factor')
    plt.ylabel('time (s)')

    plt.legend(loc=9, bbox_to_anchor=(1.25,1))
    plt.savefig(os.path.join('plots', str(batch) + '.png'), bbox_inches='tight')




