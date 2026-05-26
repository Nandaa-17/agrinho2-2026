import json
import streamlit as st
from PIL import Image
import io
import model

st.set_page_config(page_title="Classificador de Doenças em Folhas", layout='centered')

st.title("Classificador de Doenças em Folhas")
st.write("Envie uma foto da folha (soja, tomate, etc.). O sistema indica se está saudável e sugere tratamento biológico.")

uploaded = st.file_uploader("Envie a imagem da folha", type=["jpg","jpeg","png"]) 
if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption="Imagem enviada", use_column_width=True)
    with st.spinner("Analisando..."):
        result = model.detect_leaf(image)

    st.markdown(f"**Resultado:** {result['label']}")
    st.markdown(f"**Confiança:** {result['confidence']:.2f}")
    st.markdown("**Recomendação biológica:**")
    st.write(result['recommendation'])

    # botão para baixar resultado
    buffer = io.BytesIO()
    json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8')
    st.download_button("Baixar resultado (JSON)", data=json_bytes, file_name='resultado.json', mime='application/json')

else:
    st.info("Carregue uma imagem para iniciar a análise.")
