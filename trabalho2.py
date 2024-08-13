import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('i:/Downloads/train.csv')

def calculoMedia(df, column):
    valorMedia = df[column].mean()
    print(f"\nMédia de {column}: {valorMedia}")
    return valorMedia

def calculoMediana(df, column):
    valorMediana = df[column].median()
    print(f"\nMediana de {column}: {valorMediana}")
    return valorMediana

def calculoModa(df, column):
    valorModa = df[column].mode()[0]
    print(f"\nModa de {column}: {valorModa}")
    return valorModa

def calculoDesvioPadrao(df, column):
    valorDesvioPadrao = df[column].std()
    print(f"\nDesvio padrão de {column}: {valorDesvioPadrao}")
    return valorDesvioPadrao

def calculoVariancia(df, column):
    valorVariancia = df[column].var()
    print(f"\nVariância de {column}: {valorVariancia}")
    return valorVariancia

def calculoDesvioMedioAbsoluto(df, column):
    valorMedia = df[column].mean()
    valorDesvioMedioAbsoluto = (df[column] - valorMedia).abs().mean()
    print(f"\nDesvio médio absoluto de {column}: {valorDesvioMedioAbsoluto}")
    return valorDesvioMedioAbsoluto

def calculoAmplitude(df, column):
    valorAmplitude = df[column].max() - df[column].min()
    print(f"\nAmplitude de {column}: {valorAmplitude}")
    return valorAmplitude

def calculaCoeficienteVariacao(df, column):
    valorMedia = df[column].mean()
    valorDesvioPadrao = df[column].std()
    valorCoeficienteVariacao = (valorDesvioPadrao / valorMedia) * 100 if valorMedia != 0 else np.nan
    print(f"\nCoeficiente de variação de {column}: {valorCoeficienteVariacao}%")
    return valorCoeficienteVariacao

def calculoQuadrantes(df, column):
    quadrantes = df[column].quantile([0.25, 0.50, 0.75])
    print(f"\nQuadrantes de {column}:")
    print(f"1º Quartil (25%): {quadrantes[0.25]}")
    print(f"Mediana (50%): {quadrantes[0.50]}")
    print(f"3º Quartil (75%): {quadrantes[0.75]}")
    return quadrantes

def plot_statistics(df, column):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribuição de {column}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot de {column}')
    plt.tight_layout()
    plt.show()

    # Mostrar as estatísticas em uma tabela
    stats = {
        'Média': [calculoMedia(df, column)],
        'Mediana': [calculoMediana(df, column)],
        'Moda': [calculoModa(df, column)],
        'Variância': [calculoVariancia(df, column)],
        'Desvio Padrão': [calculoDesvioPadrao(df, column)],
        'Desvio Médio Absoluto': [calculoDesvioMedioAbsoluto(df, column)],
        'Amplitude': [calculoAmplitude(df, column)],
        'Coeficiente de Variação (%)': [calculaCoeficienteVariacao(df, column)],
        '1º Quartil': [calculoQuadrantes(df, column)[0.25]],
        'Mediana (50%)': [calculoQuadrantes(df, column)[0.50]],
        '3º Quartil': [calculoQuadrantes(df, column)[0.75]]
    }
    
    stats_df = pd.DataFrame(stats)
    print("\nTabela de Estatísticas:")
    print(stats_df)

plot_statistics(train_df, 'Age')