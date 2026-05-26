import os
import json
import numpy as np
from PIL import Image
import cv2


def _load_treatments():
    p = os.path.join(os.path.dirname(__file__), 'treatments.json')
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


treatments = _load_treatments()


def detect_leaf(image_pil: Image.Image) -> dict:
    """Detecta se a folha está saudável ou doente.
    Fluxo:
    - Se houver `model.h5` no diretório, tenta usar um modelo Keras para previsão.
    - Senão, aplica um heurístico rápido baseado na proporção de verde.
    Retorna dict com `label`, `confidence` e `recommendation`.
    """
    # 1) Tentativa de carregar modelo treinado (opcional)
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    try:
        if os.path.exists(model_path):
            from tensorflow.keras.models import load_model
            keras_model = load_model(model_path)
            img = image_pil.resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, 0)
            pred = keras_model.predict(arr)[0]
            # interpreta predição (suporta binário ou multi)
            if pred.shape[0] == 2:
                # assume [healthy_prob, diseased_prob]
                idx = int(pred[0] < pred[1])
                labels = ['healthy', 'diseased']
                label = labels[idx]
                confidence = float(pred[idx])
            else:
                idx = int(np.argmax(pred))
                # labels desconhecidos — usa índices
                label = f'class_{idx}'
                confidence = float(pred[idx])
            rec = treatments.get(label, {}).get('recommendation', '')
            return {'label': label, 'confidence': confidence, 'recommendation': rec}
    except Exception:
        pass

    # 2) Heurístico simples: proporção de verde
    arr = np.array(image_pil)
    if arr.size == 0:
        return {'label': 'error', 'confidence': 0.0, 'recommendation': 'Imagem inválida.'}

    mean = arr.mean(axis=(0, 1))
    r, g, b = mean[0], mean[1], mean[2]
    green_ratio = float(g / (r + g + b + 1e-8))

    if green_ratio > 0.36:
        label = 'healthy'
        confidence = green_ratio
    else:
        # detectar manchas marrons/vermelhas simples
        try:
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            # máscara para tons marrons/vermelhos (heurística)
            lower = np.array([5, 50, 20])
            upper = np.array([30, 255, 200])
            mask = cv2.inRange(hsv, lower, upper)
            brown_ratio = (mask > 0).mean()
        except Exception:
            brown_ratio = 0.0

        if brown_ratio > 0.04:
            label = 'possible_fungal'
            confidence = 0.6
        else:
            label = 'possible_nutrient_deficiency'
            confidence = 0.5

    rec = treatments.get(label, {}).get('recommendation', treatments.get('diseased', {}).get('recommendation', 'Sem recomendação.'))
    return {'label': label, 'confidence': float(confidence), 'recommendation': rec}
