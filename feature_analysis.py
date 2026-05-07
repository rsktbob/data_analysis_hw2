import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
import warnings

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

def main():
    print("="*50)
    print("Pokemon Feature Analysis & Selection")
    print("="*50)

    # 1. Load Data
    df = pd.read_csv('merged_combats.csv')
    pokemon = pd.read_csv('pokemon.csv')

    # Target: 1 if First won, 0 if Second won
    df['Target'] = (df['Winner'] == df['First_pokemon']).astype(int)

    # ==========================================
    # PART 1: Type Analysis (Single vs Dual)
    # ==========================================
    print("\n[Part 1] Type Analysis (Single vs Dual Type Win Rates)")
    
    # Count how many types a pokemon has
    pokemon['Type_Count'] = pokemon['Type 2'].notna().astype(int) + 1
    
    # Calculate average win rate for all pokemons
    wins = df['Winner'].value_counts()
    battles = df['First_pokemon'].value_counts() + df['Second_pokemon'].value_counts()
    win_rates = (wins / battles).fillna(0)
    pokemon['Win_Rate'] = pokemon['#'].map(win_rates)
    
    single_win_rate = pokemon[pokemon['Type_Count'] == 1]['Win_Rate'].mean()
    dual_win_rate = pokemon[pokemon['Type_Count'] == 2]['Win_Rate'].mean()
    
    print(f"Single Type Average Win Rate: {single_win_rate:.4f}")
    print(f"Dual Type Average Win Rate:   {dual_win_rate:.4f}")
    print(f"-> Dual Type Advantage:       {(dual_win_rate - single_win_rate)*100:.2f}%")

    # ==========================================
    # PART 2: Numerical Stats Analysis (Mean, Std & Normalization)
    # ==========================================
    print("\n[Part 2] Numerical Stats Analysis")
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    
    # Print Raw Mean and Std Dev for all Pokemons
    print("Raw Mean and Standard Deviation of Base Stats (All Pokemons):")
    raw_stats = pokemon[stats].agg(['mean', 'std']).T
    print(raw_stats)
    
    # Extract only the first pokemon's stats for this specific analysis
    first_stats = df[[f'First_{s}' for s in stats]].copy()
    
    # Apply Normalization (Z-score standardization) to put them on the same scale
    scaler = StandardScaler()
    first_stats_normalized = pd.DataFrame(scaler.fit_transform(first_stats), columns=stats)
    first_stats_normalized['Target'] = df['Target']
    
    # Calculate Mean of Normalized Stats for Winners vs Losers
    winners_norm = first_stats_normalized[first_stats_normalized['Target'] == 1][stats].mean()
    losers_norm = first_stats_normalized[first_stats_normalized['Target'] == 0][stats].mean()
    
    # Calculate the difference in standard deviations
    diff_norm = (winners_norm - losers_norm).sort_values(ascending=False)
    
    print("Difference in Standard Deviations (Z-score) between Winners and Losers:")
    for stat, val in diff_norm.items():
        print(f"  {stat:10s} : +{val:.4f} std")

    # ==========================================
    # PART 3: Filter Methods on Features (Raw vs Diff)
    # ==========================================
    print("\n[Part 3] Feature Importance (Filter Methods)")
    
    # Boolean to int
    df['First_Legendary'] = df['First_Legendary'].astype(int)
    df['Second_Legendary'] = df['Second_Legendary'].astype(int)
    
    # Create difference features
    for stat in stats + ['Legendary']:
        df[f'Diff_{stat}'] = df[f'First_{stat}'] - df[f'Second_{stat}']
    
    numerical_cols = [f'First_{s}' for s in stats] + [f'Second_{s}' for s in stats] + [f'Diff_{s}' for s in stats] + ['Diff_Legendary']
    
    X = df[numerical_cols].fillna(0)
    y = df['Target']
    
    # 1. Pearson Correlation
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print("\nTop 5 Features by Absolute Pearson Correlation:")
    print(correlations.head(5).to_string())

    # 2. ANOVA F-value
    f_values, _ = f_classif(X, y)
    f_series = pd.Series(f_values, index=X.columns).sort_values(ascending=False)
    print("\nTop 5 Features by ANOVA F-value:")
    print(f_series.head(5).to_string())
    
    print("\nAnalysis Complete.")
    print("="*50)

if __name__ == "__main__":
    main()
