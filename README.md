# agrinho2-2026

Projeto: Classificador de Doenças em Folhas por Imagem

Visão geral:
- Upload de foto de folha (soja, tomate, etc.)
- Modelo IA detecta se a folha está saudável ou doente
- Sugestões de tratamento biológico para reduzir uso de pesticidas químicos

Como rodar (ambiente com Python 3.8+):

1. Instale dependências:
```
pip install -r requirements.txt
```

2. Rode a interface Streamlit:
```
streamlit run app.py
```

Observações:
- Se houver um modelo Keras salvo em `model.h5` na raiz do projeto, o `model.py` tentará carregá-lo para inferência. Caso contrário, usa um heurístico simples baseado na cor para exemplo funcional.
- As recomendações biológicas estão em `treatments.json`.
