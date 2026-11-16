Tabla 7. Estudio de eventos: entrada de un modelo al top 50 y tokens de los modelos establecidos del mismo nido frente al otro nido

| especificación                    |   β post (mismo nido) |   ee post |   p post |   β pre (tendencia previa) |   ee pre |   p pre |   eventos |   obs |   cambio % post |
|:----------------------------------|----------------------:|----------:|---------:|---------------------------:|---------:|--------:|----------:|------:|----------------:|
| Base (todos los entrantes)        |               -0.0284 |    0.0757 |   0.7076 |                    -0.055  |   0.0517 |  0.287  |         7 |  2157 |         -2.8004 |
| Solo entrantes de pesos abiertos  |                0.1736 |    0.1238 |   0.1608 |                    -0.148  |   0.1034 |  0.1523 |         3 |   916 |         18.956  |
| Solo entrantes de pesos cerrados  |               -0.1591 |    0.1334 |   0.233  |                     0.0022 |   0.0776 |  0.9778 |         4 |  1241 |        -14.7111 |
| Entrante en top 5                 |                0.0902 |    0.0731 |   0.217  |                    -0.0036 |   0.0507 |  0.9436 |         9 |  2785 |          9.4422 |
| Entrante en top 20                |                0.0526 |    0.1168 |   0.6521 |                     0.0889 |   0.0808 |  0.2716 |         5 |  1460 |          5.4058 |
| Sin excluir eventos cercanos      |                0.0052 |    0.0243 |   0.8296 |                    -0.0094 |   0.0174 |  0.5906 |        75 | 21724 |          0.5244 |
| Incluye modelos del mismo creador |               -0.0694 |    0.0655 |   0.2894 |                    -0.0466 |   0.0478 |  0.3292 |         7 |  2398 |         -6.7    |

Nota: variable dependiente ln tokens semanales del modelo establecido. Efectos fijos de evento×modelo y evento×semana; errores agrupados por modelo. 'β post' = diferencia promedio en semanas 0 a 8 entre establecidos del mismo nido que el entrante y del otro nido; H2 predice β post < 0. 'β pre' compara semanas −4 a −2 contra −1 y debe ser ≈ 0. Se excluyen del grupo de comparación los modelos del mismo creador que el entrante.
Fuente: elaboración propia con datos de OpenRouter (openrouter.ai/rankings) y LMArena.
