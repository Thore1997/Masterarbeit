import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('weight-height.csv')
# Changing the units in cm and kg
df['Height_cm'] = df['Height'] * 2.54
df['Weight_kg'] = df['Weight'] * 0.453592

#creating imagined instance as outlier
rnd_person = {'Gender': 'Male', 'Height_cm':180, 'Weight_kg': 50}
df_rnd_person = pd.DataFrame([rnd_person])

#creating dataframe for the first 5 man and women
man = df[df['Gender'] == 'Male'].head(5)
df_man = pd.DataFrame(man)
women = df[df['Gender'] == 'Female'].head(5)
df_women = pd.DataFrame(women)

#combining dataframes into one
df_new = pd.concat([df_rnd_person, df_man, df_women], ignore_index=True)
df_new = df_new.drop(columns=['Height', 'Weight'])


#calculating euklidean distances from outlier to all the other instances
x0 = df_new.loc[0, 'Height_cm']
y0 = df_new.loc[0, 'Weight_kg']

df_new['Distance'] = np.sqrt(
    (df_new['Height_cm'] - x0)**2 +
    (df_new['Weight_kg'] - y0)**2
)
print(df_new.sort_values(by='Distance'))
nearest = df_new.sort_values(by='Distance').iloc[1:4]

plt.figure(figsize = (10,6))
plt.scatter(df_new['Height_cm'], df_new['Weight_kg'])

for i, row in nearest.iterrows():
    plt.plot([x0, row ['Height_cm']], [y0, row ['Weight_kg']],
    color = 'lightgray',
    linestyle = '--',
    linewidth = 1,
    alpha = 0.8,
    zorder = 1)

plt.scatter(df_man['Height_cm'], df_man['Weight_kg'],
            color='royalblue', label='Männer', s=50)

plt.scatter(df_women['Height_cm'], df_women['Weight_kg'],
            color='darkorange', label='Frauen', s=50)

plt.scatter(df_rnd_person['Height_cm'], df_rnd_person['Weight_kg'],
            color='green', label='Eigene Person', s=80,)


plt.xlabel('Height in cm')
plt.ylabel ('Weight in kg')
plt.savefig('Height_weight.png')

