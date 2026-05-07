import pandas as pd
import csv

# Read the CSV files
pokemon = pd.read_csv('pokemon.csv')
combats = pd.read_csv('combats.csv')

# Rename columns in pokemon to avoid confusion during merge
# We will use prefixes 'First_' and 'Second_' for the respective pokemons
pokemon_first = pokemon.add_prefix('First_')
pokemon_second = pokemon.add_prefix('Second_')

# Merge for the first pokemon
merged = combats.merge(pokemon_first, left_on='First_pokemon', right_on='First_#', how='left')

# Merge for the second pokemon
merged = merged.merge(pokemon_second, left_on='Second_pokemon', right_on='Second_#', how='left')

# Drop the redundant # columns if needed, but the user asked to add the columns.
# Usually we keep them or drop them depending on preference. 
# The user said "將對上pokemon編號的那一列加到combats"
# I will drop the duplicate ID columns 'First_#' and 'Second_#' as they are same as 'First_pokemon' and 'Second_pokemon'

merged = merged.drop(['First_#', 'Second_#'], axis=1)

# Save the merged file
merged.to_csv('merged_data.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

print("Merged file saved as merged_data.csv")
print(f"Columns: {merged.columns.tolist()}")
