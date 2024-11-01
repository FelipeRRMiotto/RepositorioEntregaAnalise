import pickle
import pandas as pd

caminho_arq_tratamento = "C:\\Users\\felipemiotto-ieg\\OneDrive - Instituto Germinare\\2024\\python\\Intesdisciplinar\\tratamento_dados.pkl"

caminho_arq_modelo = "C:\\Users\\felipemiotto-ieg\\OneDrive - Instituto Germinare\\2024\\python\\Intesdisciplinar\\modelo.pkl"

with open(caminho_arq_tratamento, 'rb') as f:
    tratamento_dados = pickle.load(f)

with open(caminho_arq_modelo, 'rb') as f:
    modelo = pickle.load(f)

colunas = ['Qual é seu genero?','Qual é a sua idade?','Qual é a média da sua renda familiar mensal?','Qual estado você mora?','Você usa muitos eletrônicos durante o dia? (6h ou mais)','Qual grau de educação você tem?','Você pratica esportes? (pelo menos 3 vezes na semana)','Você frequenta muito espaços públicos? (parques, museus e etc)']

dados = [['Masculino',16,'Maior que R$15000','SP','v','Pós graduação','f','f']]

dado_teste = pd.DataFrame(dados,columns=colunas)

x = pd.DataFrame(tratamento_dados.transform(dado_teste),columns=tratamento_dados.get_feature_names_out())

previsao = modelo.predict(x)

if previsao == 1:
    print("cliente em potencial")
else:
    print("usuario comum")