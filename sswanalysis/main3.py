# Postprocess results
from datetime import datetime as dt

import numpy as np
import pandas as pd

datestype1 = ['1973-02-04', '1973-02-05', '1973-02-06', '1973-02-07', '1973-02-09',
              '1987-02-08', '1987-02-09', '1987-02-10', '1987-02-11', '1987-02-15',
              '1987-02-16', '2004-01-08', '2004-01-09', '2004-01-10', '2004-01-11',
              '2004-01-12', '2006-01-28', '2006-01-29', '2006-01-30', '2006-01-31',
              '2006-02-01', '2006-02-02', '2006-02-03', '2006-02-04', '2006-02-05',
              '2006-02-06', '2006-02-07', '2006-02-08', '2006-02-09', '2006-02-10',
              '2006-02-11', '2006-02-12', '2006-02-13', '2006-02-14', '2006-02-15',
              '2006-02-16', '2006-02-17', '2009-02-08', '2013-01-21', '2013-01-24',
              '2013-01-26']

datestype2 = ['1958-01-30', '1963-02-17', '1966-02-27', '1968-01-08', '1970-12-26',
              '1971-01-22', '1973-02-08', '1979-02-21', '1979-02-22', '1979-02-23',
              '1979-02-24', '1979-02-25', '1984-12-30', '1985-01-20', '1987-02-17',
              '1987-02-27', '1987-12-17', '1987-12-18', '1989-02-19', '1989-02-20',
              '1989-02-21', '1998-12-20', '2003-02-18', '2003-02-19', '2006-02-21',
              '2006-02-22', '2006-02-23', '2006-02-24', '2006-02-25', '2009-01-26',
              '2009-01-27', '2009-01-28', '2009-01-29', '2009-02-02', '2009-02-03',
              '2009-02-04', '2013-01-08', '2013-01-14', '2013-01-16', '2013-01-17',
              '2013-01-18', '2013-01-19', '2013-01-20']

type1array = np.ones_like(datestype1)
type2array0 = np.ones_like(datestype2)
type2array = []
for i in type2array0:
    type2array.append(2)

type2array = np.asarray(type2array)

dates = np.concatenate((datestype1, datestype2), axis=0)
types = np.concatenate((type1array, type2array), axis=0)
datesaux = [dt.strptime(date, '%Y-%m-%d').date() for date in dates]

datesfmt = []
for i in datesaux:
    datesfmt.append(dt.strftime(i, format=' %d %b %Y'))

df = pd.DataFrame({'Dates': dates, 'Datesfmt': datesfmt, 'Type': types})
df['Aux'] = "1"

df['Dates'] = pd.to_datetime(df['Dates'])
df = df.sort_values(by='Dates')

df['Type'] = pd.to_numeric(df['Type'])
df['year'] = pd.DatetimeIndex(df['Dates']).year

df = df.drop_duplicates(subset=['year'], keep='first')

df1 = df[df['Type'] == 1].drop(columns=['Type', 'Dates', 'year'])
df2 = df[df['Type'] == 2].drop(columns=['Type', 'Dates', 'year'])

# print(df.to_latex(index=False))
print(df1.to_latex(index=False))
print(df2.to_latex(index=False))
