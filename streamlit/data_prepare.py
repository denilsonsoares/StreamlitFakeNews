import pandas as pd
def prepare_data(df_fake, df_true):
    # Adicionar uma coluna que indica se é Fake ou True
    df_fake['label'] = 'Fake'
    df_true['label'] = 'True'

    # Manter apenas registros com datas válidas
    df_fake['date'] = pd.to_datetime(df_fake['date'], errors='coerce')
    df_true['date'] = pd.to_datetime(df_true['date'], errors='coerce')

    # Filtrar apenas registros com datas não nulas
    df_fake = df_fake.dropna(subset=['date'])
    df_true = df_true.dropna(subset=['date'])

    # Combinar os dois DataFrames em um único DataFrame
    df_combined = pd.concat([df_fake, df_true], ignore_index=True)

    # Adicionar um identificador único para cada notícia
    df_combined.reset_index(inplace=True)
    df_combined.rename(columns={'index': 'id'}, inplace=True)

    return df_combined