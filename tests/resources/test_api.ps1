# test_api_ia.ps1
# Script para probar la API de detección IA vs humano en http://localhost:8001/predict

# URL de la API (cámbiala si usas otro puerto/host)
$apiUrl = "http://localhost:8001/predict"

# Texto a evaluar (puedes modificarlo libremente)
$text = @"
En los últimos años, el uso de herramientas de Inteligencia Artificial para generar textos académicos
se ha extendido de forma silenciosa pero constante. Muchos estudiantes recurren a estos sistemas para
redactar trabajos, resúmenes o incluso proyectos completos, sin reflexionar sobre las implicaciones
éticas que tiene presentar como propio un contenido generado automáticamente. Este fenómeno plantea
preguntas importantes sobre la evaluación del aprendizaje, la autoría real y la necesidad de desarrollar
métodos fiables para detectar textos producidos por modelos de lenguaje.
"@

# Construimos el JSON correctamente como objeto → JSON → bytes UTF-8
$payloadObject = @{
  text = $text
}

$payloadJson = $payloadObject | ConvertTo-Json -Depth 3
$payloadBytes = [System.Text.Encoding]::UTF8.GetBytes($payloadJson)

Write-Host "Enviando petición POST a $apiUrl..."
Write-Host "Texto (primeros 120 caracteres):" ($text.Substring(0, [Math]::Min(120, $text.Length))) "..." -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod `
        -Uri $apiUrl `
        -Method POST `
        -ContentType "application/json; charset=utf-8" `
        -Body $payloadBytes

    Write-Host "`nRespuesta de la API:" -ForegroundColor Green
    $response | Format-List
}
catch {
    Write-Host "`nError llamando a la API:" -ForegroundColor Red
    Write-Host $_
}
