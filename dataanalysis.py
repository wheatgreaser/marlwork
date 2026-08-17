import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

travel_time_1 = np.load('mean_tt_torch.npy')
df = pd.DataFrame({'travel_time': travel_time_1})[:9000]
df['rolling_av'] = df.travel_time.rolling(1000).mean()
fig, ax = plt.subplots()

ax.plot(list(range(0,(9000))), df['rolling_av'])
plt.show()
