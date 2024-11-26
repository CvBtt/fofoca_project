from data_input import import_data
from data_cleaning import datasets
from match_making import make_teams
import ortools
import numpy
import pandas

def main():
    print("""
.-----------------------------------------------------------------.
| _____      __                   ____            _           _   |
||  ___|__  / _| ___   ___ __ _  |  _ \ _ __ ___ (_) ___  ___| |_ |
|| |_ / _ \| |_ / _ \ / __/ _` | | |_) | '__/ _ \| |/ _ \/ __| __||
||  _| (_) |  _| (_) | (_| (_| | |  __/| | | (_) | |  __/ (__| |_ |
||_|  \___/|_|  \___/ \___\__,_| |_|   |_|  \___// |\___|\___|\__||
|                                              |__/               |
'-----------------------------------------------------------------' 
""")
    print(ortools.__version__, pandas.__version__, numpy.__version__)
    
    #Importando dados dos csv's
    overall_df, positions_df, coefficients_df = import_data.import_all_data()
    
    #Limpando os dataframes
    positions, conflicts, coefficients, all_players, skills, all_positions = datasets.make_dict(overall_df, positions_df, coefficients_df)
    
    #Match Making loop
    final_stats = make_teams.make_teams(positions, conflicts, coefficients, all_players, skills, all_positions)
    print(final_stats)
    
    #End Message
    print("--- Thanks for playing! ---")
    
if __name__ == '__main__':
    main()