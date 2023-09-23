# -*- coding: utf-8 -*-
#==============================================================================
# INTELIGÊNCIA ARTIFICIAL APLICADA
# REDES NEURAIS - SEMANA 5
# REDE NEURAL COMPETITIVA (RNC)
# Aluno: Fabricio Magalhães Sena
#==============================================================================

#==============================================================================
# IMPORTAÇÃO DE BIBLIOTECAS
#==============================================================================
from dataclasses import dataclass
import datetime
import secrets
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#==============================================================================
# CRIAÇÃO DA CLASSE PARA REDE NEURAL COMPETITIVA (RNC)
#==============================================================================
class RNC:
    def __init__(self, input_shape, num_neurons):
        self.input_shape = input_shape
        self.num_neurons = num_neurons
        self.weights = np.random.rand(num_neurons, input_shape)
    
    def train(self, input_data, learning_rate=0.1, num_epochs=5):
        progress_bar = tqdm(total=num_epochs, desc='Treinando', unit=' épocas', ncols=80)
        for epoch in range(num_epochs):
            for input_sample in input_data:
                winner_neuron = self.get_winner_neuron(input_sample)
                self.update_weights(input_sample, winner_neuron, learning_rate)
            progress_bar.set_postfix(epoch=epoch+1)
            progress_bar.update()
        progress_bar.close()
    
    def get_winner_neuron(self, input_sample):
        distances = np.linalg.norm(input_sample - self.weights, axis=1)
        return np.argmin(distances)
    
    def update_weights(self, input_sample, winner_neuron, learning_rate):
        self.weights[winner_neuron] += learning_rate * (input_sample - self.weights[winner_neuron])
        
    def predict(self, input_data):
        predictions = []
        for input_sample in input_data:
            winner_neuron = self.get_winner_neuron(input_sample)
            predictions.append(winner_neuron)
        return predictions


@dataclass
class ConcurrentNeuralNetworkTestResult:
    function_str: str = None
    cilindrada_info: str = None
    previsao_y: str = None
    model_trained: RNC = None
    image_name: str = None

@dataclass
class ConcurrentNeuralNetworkTestParam:
    neurons_qty: int = 10,
    epochs: int = 100,
    learning_percentual: int = 2.0,
    polinomial_order: int = 1,
    cilindrada_to_predict: int = 1.0
    name: str = None


class ConcurrentNeuralNetworkTest:
    def __init__(self):
        pass

    def prepare_dataset(self):
        self.data = pd.read_csv('./data/dataset.csv')
        #==============================================================================
        # CARREGAMENTO DA BASE DE DADOS E SELEÇÃO DAS COLUNAS DE INTERESSE
        #==============================================================================

        # Carregar o conjunto de dados "base_veiculos.csv"
        data = self.data

        # Extrair as colunas de interesse
        columns = ['Cilindrada', 'Eficiencia']
        data = data[columns]

        #==============================================================================
        # CONFIGURAÇÃO DA BASE DE DADOS
        #==============================================================================

        # Remover linhas com valores em branco ou zero
        data = data.dropna()
        data = data[(data != 0).all(axis=1)]

    def set_config_params(self, params = ConcurrentNeuralNetworkTestParam()):
        #==============================================================================
        # CONFIGURAÇÃO DA ESTRUTURA DA REDE NEURAL COMPETITIVA
        #==============================================================================

        # DEFINA A QUANTIDADE DE NEURÔNIOS DA REDE
        self.neurons_qty = params.neurons_qty

        # DEFINA O NÚMERO DE ÉPOCAS PARA TREINAMENTO DA REDE
        self.epochs = params.epochs

        # DEFINA A TAXA DE APRENDIZADO DA REDE
        self.learning_percentual = params.learning_percentual

        # Ordem do Polinômio para Regressão cilindrada x efciência
        self.polinomial_order = params.polinomial_order

        # Informe um valor de Cilindrada (em L) para prever a Eficiência (em Km/L)
        self.cilindrada_to_predict = params.cilindrada_to_predict

        self.test_id = params.name + '-' + secrets.token_hex(8)


    def run(self):
        # Converter as colunas em arrays numpy
        cilindrada = self.data['Cilindrada'].values
        eficiencia = self.data['Eficiencia'].values

        #------------------------------------------------------------------------------
        # Concatenar os dados combinados em um único array

        # Cilindrada x Eficiência
        combinacao = np.column_stack((cilindrada, eficiencia))
        inshape = combinacao.shape[1]

        #==============================================================================
        # CRIAÇÃO DA REDE NEURAL COMPETITIVA
        #==============================================================================

        # Criação da Rede Neural Competitiva 1
        rnc = RNC(input_shape=inshape, num_neurons=self.neurons_qty)

        #==============================================================================
        # TREINAMENTO DA REDE NEURAL COMPETITIVA
        #==============================================================================

        # Treinamento da Rede Neural Competitiva 1
        rnc.train(combinacao, learning_rate=self.learning_percentual, num_epochs=self.epochs)

        #==============================================================================
        # REALIZAR PREDIÇÕES DA REDE NEURAL COMPETITIVA
        #==============================================================================

        # Realizar a predição cilindrada x kmpl
        predictions = rnc.predict(combinacao)

        #==============================================================================
        # CALCULAR OS POLINÔMIOS DAS REGRESSÕES
        #==============================================================================

        # Cálculo do polinômio para Cilindrada X Eficiência
        coefficients = np.polyfit(cilindrada, eficiencia, self.polinomial_order)
        polynomial = np.poly1d(coefficients)

        #==============================================================================
        # GERAR A TABELA DE AGRUPAMENTOS (CLUSTERS)
        #==============================================================================

        #------------------------------------------------------------------------------
        # Tabela Cilindrada X Eficiência

        table_data = {'Cil.': cilindrada,
                    'Efic.': eficiencia,
                    'Group': predictions}

        df_table = pd.DataFrame(table_data)

        grouped_table = df_table.groupby('Group').agg({'Cil.': ['min', 'max'],
                                                    'Efic.': ['min', 'max'],
                                                    'Group': 'size'})

        grouped_table.columns = ['Cil. (min)', 'Cil. (max)', 'Efic. (min)',
                                'Efic. (max)', 'Elementos']

        # Formatar valores reais para duas casas decimais
        grouped_table = grouped_table.round(decimals=2)

        #==============================================================================
        # REALIZAR AS PREVISÕES DA REGRESSÃO
        #==============================================================================

        #------------------------------------------------------------------------------
        # Prever o valor de Eficiência correspondente a Cilindrada informada
        previsao_y = polynomial(self.cilindrada_to_predict)
        previsao_y = '{:.2f}'.format(previsao_y)

        #==============================================================================
        # PLOTAR OS RESULTADOS EM GRÁFICOS DE DISPERSÃO
        #==============================================================================

        #------------------------------------------------------------------------------
        # Cilindrada (L) X Eficiência (Km/L)

        # Obter os termos do polinômio formatados
        terms = []

        for i, coeff in enumerate(polynomial.coeffs):
            power = polynomial.order - i
            term = f"{coeff:.2f}x^{power}" if power > 1 else f"{coeff:.2f}x" if power == 1 else f"{coeff:.2f}"
            terms.append(term)

        # Construir a string da função f(x)
        function_str = "f(x) = " + " + ".join(terms)

        # Plotar o gráfico de dispersão
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(cilindrada, eficiencia, c=predictions)
        ax.set_xlabel("Cilindrada (L)")
        ax.set_ylabel("Eficiência (Km/L)")
        ax.set_title("Cilindrada X Eficiência")
        ax.set_xticks(np.arange(0, np.max(cilindrada) + 2, 1))
        ax.set_xticklabels(['%.1f' % x for x in np.arange(0, np.max(cilindrada) + 2, 1)])
        ax.grid(linestyle='dotted')

        # Plotar a linha de regressão
        x_values = np.linspace(np.min(cilindrada), np.max(cilindrada), 100)
        y_values = polynomial(x_values)
        ax.plot(x_values, y_values, color='red')

        image_hash = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        image_name = f'./output/v3/{self.test_id}-{image_hash}-{secrets.token_urlsafe(4)}.png'
        plt.savefig(image_name)

        #==============================================================================
        # APRESENTAR DAS TABELAS COM OS CLUSTERS
        #==============================================================================

        print('\n' + '=' * 70)
        print('TABELA DE AGRUPAMENTO (CLUSTERS)')
        print('=' * 70)
        print('\n')

        #------------------------------------------------------------------------------
        # Tabela Cilindrada (L) X Eficiência (Km/L)
        print("Tabela 1. Cilindrada (L) X Eficiência (Km/L)")
        print('_' * 70)
        print(grouped_table)
        print('_' * 70)
        print('\n' * 2)

        #==============================================================================
        # APRESENTAR AS PREVISÕES DAS REGRESSÕES
        #==============================================================================

        print('\n' + '=' * 70)
        print('PREVISÃO DA REGRESSÃO')
        print('=' * 70)
        print('\n')

        #------------------------------------------------------------------------------
        # Imprimir resultados da função f(x) para Cilindrada X Eficiência
        print('Função polinomial para Cilindrada X Eficiência')
        print('_' * 70)
        print(function_str + '\n')  
        print("Cilindrada informada:", self.cilindrada_to_predict, "L")
        print("Eficiência prevista:", previsao_y, "Km/L")
        print('_' * 70)
        print('\n')

        #==============================================================================

        return ConcurrentNeuralNetworkTestResult(
            function_str=function_str,
            cilindrada_info=self.cilindrada_to_predict,
            previsao_y=previsao_y,
            model_trained=rnc,
            image_name=image_name
        )


    def save_results_on_csv(self, results: List[Dict]):
        df = pd.DataFrame(results)
        csv_hash = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        df.to_csv(f'./results/results-{csv_hash}.csv', index=False)


possible_config_combinations = []


for epochs_qty in [10, 20, 30, 40, 50, 60, 70]:
    possible_config_combinations.append(
    ConcurrentNeuralNetworkTestParam(
        neurons_qty=2,
        epochs=epochs_qty,
        learning_percentual=0.9,
        polinomial_order=1,
        cilindrada_to_predict=50,
        name='test-epoch-' + str(epochs_qty)
    ))


for neurons_qty in range(2, 20):
    possible_config_combinations.append(
    ConcurrentNeuralNetworkTestParam(
        neurons_qty=neurons_qty,
        epochs=40,
        learning_percentual=0.9,
        polinomial_order=1,
        cilindrada_to_predict=8.0,
        name='test-neurons-' + str(neurons_qty)
    ))


for polinomial_orderinomial in range(2, 20):
    possible_config_combinations.append(
    ConcurrentNeuralNetworkTestParam(
        neurons_qty=15,
        epochs=40,
        learning_percentual=0.9,
        polinomial_order=polinomial_orderinomial,
        cilindrada_to_predict=8.0,
        name='polinomial-order-' + str(polinomial_orderinomial)
    ))


possible_config_combinations.append(
    ConcurrentNeuralNetworkTestParam(
        neurons_qty=20,
        epochs=30,
        learning_percentual=0.8,
        polinomial_order=3,
        cilindrada_to_predict=14.0,
        name='final-3'
    )
)


test = ConcurrentNeuralNetworkTest()
results = []

for param in possible_config_combinations:
    test.prepare_dataset()
    test.set_config_params(param)
    results.append(test.run())

merge_between_params_and_results = []

for index in range(0, len(results) - 1):
    merged = results[index].__dict__.copy()
    merged.update(possible_config_combinations[index].__dict__)
    merge_between_params_and_results.append(merged)

test.save_results_on_csv(merge_between_params_and_results)