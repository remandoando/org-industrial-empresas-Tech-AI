Tabla 6. Robustez del logit anidado VI

| especificación                                                |      σ |   ee(σ) |        α |   ee(α) |   obs |   cerrado → cerrado |   cerrado → abierto |   cerrado: mismo nido entre internos |   F mín. primera etapa |
|:--------------------------------------------------------------|-------:|--------:|---------:|--------:|------:|--------------------:|--------------------:|-------------------------------------:|-----------------------:|
| Mercado potencial λ = 1.2                                     | 0.6877 |  0.1899 |  -7.7182 |  7.1153 |  3546 |              0.7922 |              0.0563 |                               0.9345 |                 5.265  |
| Mercado potencial λ = 1.5                                     | 0.6877 |  0.1899 |  -7.7182 |  7.1153 |  3546 |              0.7626 |              0.0449 |                               0.9451 |                 5.265  |
| Mercado potencial λ = 2.0                                     | 0.6877 |  0.1899 |  -7.7182 |  7.1153 |  3546 |              0.7334 |              0.0335 |                               0.9567 |                 5.265  |
| Mercado potencial λ = 3.0                                     | 0.6877 |  0.1899 |  -7.7182 |  7.1153 |  3546 |              0.7046 |              0.0222 |                               0.9696 |                 5.265  |
| Incluye variantes gratuitas                                   | 0.6802 |  0.1888 |  -8.1324 |  6.9821 |  4106 |              0.7705 |              0.059  |                               0.9292 |                 4.4519 |
| Sin las primeras 4 semanas de cada modelo                     | 0.0823 |  0.3171 |   2.8976 |  5.4004 |  2832 |              0.3113 |              0.082  |                               0.7915 |                 6.7396 |
| Solo calidad observada (sin imputar)                          | 0.7708 |  0.372  |  -3.9778 |  7.0873 |  2693 |              0.7893 |              0.0307 |                               0.9628 |                 6.5302 |
| Sin ratings de variantes 'max' (11 modelos, calidad imputada) | 0.9767 |  0.3461 | -13.1426 | 11.4679 |  3546 |              0.9815 |              0.0035 |                               0.9965 |                 4.9    |
| Instrumentos: solo calidad                                    | 1.1413 |  0.353  | -12.0304 | 10.3412 |  3546 |              1.1158 |             -0.0217 |                               1.0199 |                 4.8693 |
| Instrumentos: solo conteos                                    | 1.0273 |  1.1791 | -13.3323 | 29.344  |  3546 |              1.0219 |             -0.0041 |                               1.004  |                 3.3273 |
| Errores agrupados por creador                                 | 0.6877 |  0.1287 |  -7.7182 |  7.5709 |  3546 |              0.7626 |              0.0449 |                               0.9451 |                 9.6597 |
| Nidos por tercil de calidad                                   | 0.8351 |  0.1922 |   8.712  |  5.6614 |  3546 |            nan      |            nan      |                             nan      |                 4.1169 |

Nota: "cerrado → cerrado" y "cerrado → abierto" son la fracción promedio de los tokens que pierde un modelo cerrado al subir su precio que se va a otro cerrado o a uno abierto. Con efectos fijos de semana, λ solo mueve ln s_0t, que queda absorbido: α y σ no cambian y solo cambian elasticidades y desvíos. En la fila de nidos por tercil σ mide sustitución dentro del tercil. En la fila sin variantes 'max', la calidad de los modelos cuyo rating de Arena viene de su variante de máximo esfuerzo se reemplaza por la mediana del nido en la semana.
Fuente: elaboración propia con datos de OpenRouter (openrouter.ai/rankings) y LMArena.
