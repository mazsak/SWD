import matplotlib.pyplot as plt
import pandas as pd

with open('temp.txt') as f:
    slownik = eval(f.readline())
df = pd.DataFrame(slownik)
print(df)
plt.plot(df)
plt.legend(df.columns)
plt.show()