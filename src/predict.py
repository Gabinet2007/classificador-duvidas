import joblib

# Carrega o modelo treinado salvo anteriormente
modelo = joblib.load('../modelo_duvidas.pkl')

# Loop para testar dúvidas
while True:
    pergunta = input("Digite sua dúvida (ou 'sair' para encerrar): ")
    if pergunta.lower() == 'sair':
        break
    categoria = modelo.predict([pergunta])[0]
    print(f"Categoria prevista: {categoria}\n")
