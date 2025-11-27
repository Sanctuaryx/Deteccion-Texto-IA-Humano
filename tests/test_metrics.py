import json
import requests

API_URL = "http://localhost:8001/predict"

text = """
En los últimos años, el uso de herramientas de Inteligencia Artificial para generar textos académicos
se ha extendido de forma silenciosa pero constante. Muchos estudiantes recurren a estos sistemas para
redactar trabajos, resúmenes o incluso proyectos completos, sin reflexionar sobre las implicaciones
éticas que tiene presentar como propio un contenido generado automáticamente.
"""

payload = {"text": text}

print(f"Enviando petición POST a {API_URL}...")
resp = requests.post(API_URL, json=payload, timeout=60)
print("Status:", resp.status_code)
print("Respuesta JSON:")
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
