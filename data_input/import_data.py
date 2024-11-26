import pandas as pd
import numpy as np

def import_all_data():
    overall_df = pd.read_csv("csv/overall.csv", sep=";")
    positions_df = pd.read_csv("csv/posicao_picuinha.csv", sep=";", na_values=["Null"])
    coefficients_df = pd.read_csv("csv/valores_calibracao.csv", sep=";")
    
    return overall_df, positions_df, coefficients_df

    
    

