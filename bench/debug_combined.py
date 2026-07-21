import pandas as pd
from sympy import python
p='results/combined_per_file.csv'
df=pd.read_csv(p)
print('columns', df.columns.tolist())
print(df.head(12).to_string())
for model, g in df.groupby('model'):
    print('model', model, type(g))
    for col in ['median_latency_ms','mean_latency_ms','p95','wer']:
        print(' col',col,'in columns?', col in g.columns, 'get() type:', type(g.get(col)))
    break
#python -m bench.smoke1``