import numpy as np
import pandas as pd

def make_dict(overall_df, positions_df, coefficients_df):

    positions = {}
    conflicts = {}
    coefficients = {}

    all_players = overall_df['Nome'].tolist()
    skills = overall_df.set_index('Nome').to_dict('index')
    all_positions = coefficients_df['Posicao'].tolist()
    
    for _, row in positions_df.iterrows():
        player = row['Nome']
        player_positions = row[1:6].dropna().tolist()
        positions[player] = player_positions
        conflict_str = row['Conflito']
        if pd.isna(conflict_str):
            conflicts[player] = []
        else:
            conflicts[player] = conflict_str.split('/')


    for _, row in coefficients_df.iterrows():
        position = row['Posicao']
        coefficients[position] = {
            'Saque': row['Saque'],
            'Recepcao': row['Recepcao'],
            'Levantamento': row['Levantamento'],
            'Ataque': row['Ataque'],
            'Bloqueio': row['Bloqueio'],
            'Defesa': row['Defesa']
        }
    
    return positions, conflicts, coefficients, all_players, skills, all_positions


    