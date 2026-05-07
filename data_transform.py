import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    print("Loading data...")
    df = pd.read_csv('merged_combats.csv')
    
    print("Preparing features...")
    
    # 1. Target Label (+1 for First, -1 for Second)
    y = np.where(df['Winner'] == df['First_pokemon'], 1, -1)
    
    # 2. Type Differences (18 Types)
    types = sorted(['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 
                    'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'])
    
    # Create binary matrices for types
    first_types = pd.DataFrame(0, index=np.arange(len(df)), columns=types)
    second_types = pd.DataFrame(0, index=np.arange(len(df)), columns=types)
    
    # Fill binary matrices
    for t in types:
        first_types[t] = ((df['First_Type 1'] == t) | (df['First_Type 2'] == t)).astype(int)
        second_types[t] = ((df['Second_Type 1'] == t) | (df['Second_Type 2'] == t)).astype(int)
        
    diff_types = first_types - second_types
    
    # 3. Legendary Difference
    diff_legendary = df['First_Legendary'].astype(int) - df['Second_Legendary'].astype(int)
    
    # 4. Stat Differences
    diff_speed = df['First_Speed'] - df['Second_Speed']
    diff_attack = df['First_Attack'] - df['Second_Attack']
    diff_sp_atk = df['First_Sp. Atk'] - df['Second_Sp. Atk']
    diff_sp_def = df['First_Sp. Def'] - df['Second_Sp. Def']
    diff_hp = df['First_HP'] - df['Second_HP']
    
    # Combine all features in strict order
    # Index 1-18: Type Differences
    # Index 19: Diff_Legendary
    # Index 20-24: Diff Stats
    features = []
    
    for t in types:
        features.append(diff_types[t].values)
        
    features.append(diff_legendary.values)
    features.append(diff_speed.values)
    features.append(diff_attack.values)
    features.append(diff_sp_atk.values)
    features.append(diff_sp_def.values)
    features.append(diff_hp.values)
    
    # Transpose to get rows of features
    X = np.column_stack(features)
    
    print(f"Dataset shape: {X.shape}")
    print("Writing to libsvm format (svm_data.txt)...")
    
    with open('pokemon_data.txt', 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(df))):
            line = [str(y[i])]
            for j, val in enumerate(X[i]):
                # LIBSVM sparse format: omit zero values
                if val != 0:
                    line.append(f"{j+1}:{val}")
            f.write(" ".join(line) + "\n")
            
    print("Success! Data saved to 'svm_data.txt'")
    print("Feature Mapping:")
    for idx, name in enumerate(types + ['diff_Legendary', 'diff_Speed', 'diff_Attack', 'diff_SpAtk', 'diff_SpDef', 'diff_HP']):
        print(f"  {idx+1}: {name}")

if __name__ == "__main__":
    main()