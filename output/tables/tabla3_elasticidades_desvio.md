Tabla 3. Destino de los tokens perdidos ante un alza de precio (razones de desvío), promedio ponderado por tokens

| nido de origen   | medida                          |   estimación |   IC95 inf |   IC95 sup |
|:-----------------|:--------------------------------|-------------:|-----------:|-----------:|
| abierto          | desvio_mismo_nido               |        0.738 |      0.465 |      0.974 |
| abierto          | desvio_otro_nido                |        0.081 |      0.008 |      0.165 |
| abierto          | desvio_externo                  |        0.181 |      0.018 |      0.37  |
| abierto          | mismo_nido_entre_internos       |        0.901 |      0.735 |      0.992 |
| abierto          | participacion_del_nido (tokens) |        0.665 |    nan     |    nan     |
| cerrado          | desvio_mismo_nido               |        0.763 |      0.511 |      0.977 |
| cerrado          | desvio_otro_nido                |        0.045 |      0.004 |      0.093 |
| cerrado          | desvio_externo                  |        0.193 |      0.019 |      0.396 |
| cerrado          | mismo_nido_entre_internos       |        0.945 |      0.851 |      0.996 |
| cerrado          | participacion_del_nido (tokens) |        0.335 |    nan     |    nan     |

Nota: intervalos por simulación paramétrica de σ (477 extracciones en (0, 1), semilla 20261115); los desvíos no dependen de α. No se reportan elasticidades porque el coeficiente de precio no está identificado (Tabla 2). Los tres desvíos suman 1. 'mismo_nido_entre_internos' es la fracción que se queda en el mismo nido entre los tokens que no se van al bien externo; H2 se cumple si es claramente mayor que la participación del nido.
Fuente: elaboración propia con datos de OpenRouter (openrouter.ai/rankings) y LMArena.
