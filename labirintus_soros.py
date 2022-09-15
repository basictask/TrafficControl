# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#%% dataframe letrehozasa
# n = int(input('adjon meg egy szamot: '))
n = 5
nrow = 5+(n-1)*4
ncol = 10+(n-1)*9


#%% mindegyik ertek kicserelese ures stringre es feltoltes teljessel
def get_df():
    # dict elemek felvetele
    cells = {}
    for i in np.arange(n):
        for j in np.arange(n):
            cells[(i+1, j+1)] = []
    
    df = pd.DataFrame(columns=np.arange(ncol)+1, index=np.arange(nrow)+1)
    
    for col in df.columns:
        df[col].values[:] = ""
        
    df.iloc[4::4, :] = '-'
    df.iloc[:, 9::9] = '|'
    
    df.iloc[0, :] = '-'
    df.iloc[:, 0] = '|'
    
    
    # oszlop ritkitas
    iter_i = [x for x in df.iloc[9::9, 2::4]][:n] # sorok [3, 7, 11, 15, 19, 23] n
    iter_j = [x for x in df.iloc[2::4, 9::9]][:-1] # oszlopok [10, 19, 28, 37, 46] n-1
    
    for i,x in zip(iter_i, np.arange(len(iter_i))+1):
        for j,y in zip(iter_j, np.arange(len(iter_j))+1):
            if np.random.uniform(0,1)>0.5:
                df.loc[i, j] = ''
                
                cells[(x, y)].append((x, y+1))
                cells[(x, y+1)].append((x, y))
            
    
    # sor ritkitas
    iter_i = [x for x in df.iloc[4::9, 4::4]][:n-1]  # sorok [5, 9, 13, 17, 21] n-1
    iter_j = [x for x in df.iloc[4::4, 4::9]] # oszlopok [5, 14, 23, 32, 41, 50] n
    
    for i,x in zip(iter_i, np.arange(len(iter_i))+1): 
        for j,y in zip(iter_j, np.arange(len(iter_j))+1):
            if np.random.uniform(0,1)>0.5:
                df.loc[i, j] = ''
                df.loc[i, j+1] = ''
                
                cells[(x, y)].append((x+1, y))
                cells[(x+1, y)].append((x, y))

    return df, cells


#%% vane ures a cellsben
def isempty(cells):
    lengths = set()
    for i in np.arange(n)+1:
        for j in np.arange(n)+1:
            lengths.add(len(cells[(i,j)]))
    if (0 in lengths):
        return True
    return False  


#%% legyen mindig bejarhato 
flag = True
while(flag == True):
    df, cells = get_df()
    flag = isempty(cells)
    

#%% halmaz es rekurzio
def DFS(G,v,seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths

# DFS futtatas es metrika
all_paths = DFS(cells, (1,1))
max_len   = max(len(p) for p in all_paths)
max_paths = [p for p in all_paths if len(p) == max_len]

# Output
print("Longest Paths:")
for p in max_paths: print("  ", p)
print("Longest Path Length:")
print(max_len)