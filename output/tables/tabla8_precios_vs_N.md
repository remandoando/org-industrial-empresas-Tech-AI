Tabla 8. Precio de un modelo abierto según su número de proveedores (N = 1 es la base)

| N_cat   |   mercados |   precio_min_mediana |   precio_mediano_mediana |   dispersion_cv |   con_tamano_de_mercado |
|:--------|-----------:|---------------------:|-------------------------:|----------------:|------------------------:|
| 1       |         80 |                0.372 |                    0.372 |         nan     |                       2 |
| 2       |         21 |                0.31  |                    0.346 |           0.219 |                       1 |
| 3       |          6 |                0.147 |                    0.275 |           0.171 |                       1 |
| 4       |          9 |                0.44  |                    0.523 |           0.335 |                       0 |
| 5+      |         39 |                0.388 |                    0.6   |           0.335 |                      20 |

| precio   | N                   | especificación   |   mercados |     N=2 |   ee N=2 |      N=3 |   ee N=3 |      N=4 |   ee N=4 |     N=5+ |   ee N=5+ |     N=3+ |   ee N=3+ |   p H0 Bertrand (N≥3 = N=2) |
|:---------|:--------------------|:-----------------|-----------:|--------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|---------:|----------:|----------------------------:|
| mínimo   | detallado           | (1) Base         |        155 | -0.4997 |   0.2304 |  -0.6203 |   0.3385 |  -0.5615 |   0.5639 |  -0.3828 |    0.2468 | nan      |  nan      |                      0.9258 |
| mínimo   | detallado           | (2) + parámetros |         84 | -0.537  |   0.2365 |  -0.852  |   0.3317 |  -0.417  |   0.5762 |  -0.6278 |    0.3464 | nan      |  nan      |                      0.8356 |
| mínimo   | agrupado (1, 2, 3+) | (1) Base         |        155 | -0.5012 |   0.2286 | nan      | nan      | nan      | nan      | nan      |  nan      |  -0.448  |    0.2243 |                      0.8431 |
| mínimo   | agrupado (1, 2, 3+) | (2) + parámetros |         84 | -0.5367 |   0.2337 | nan      | nan      | nan      | nan      | nan      |  nan      |  -0.6108 |    0.2734 |                      0.8119 |
| mediano  | detallado           | (1) Base         |        155 | -0.3032 |   0.2185 |  -0.3417 |   0.2853 |  -0.1885 |   0.4684 |   0.2279 |    0.229  | nan      |  nan      |                      0.1412 |
| mediano  | detallado           | (2) + parámetros |         84 | -0.3178 |   0.2438 |  -0.531  |   0.2026 |   0.0606 |   0.4001 |  -0.0075 |    0.3008 | nan      |  nan      |                      0.2194 |
| mediano  | agrupado (1, 2, 3+) | (1) Base         |        155 | -0.3068 |   0.2164 | nan      | nan      | nan      | nan      | nan      |  nan      |   0.074  |    0.2079 |                      0.1136 |
| mediano  | agrupado (1, 2, 3+) | (2) + parámetros |         84 | -0.3181 |   0.2407 | nan      | nan      | nan      | nan      | nan      |  nan      |  -0.0741 |    0.2299 |                      0.3774 |

Nota: variable dependiente ln del precio mínimo o mediano entre proveedores (USD por millón de tokens, mezcla 75/25 entrada/salida). Un mercado = un modelo de pesos abiertos. Errores robustos HC1. 'agrupado' junta N = 3, 4 y 5+ en una categoría 3+ para ganar potencia; la especificación (3) solo se estima con 40 mercados o más. 'p H0 Bertrand' prueba que los coeficientes de N ≥ 3 son iguales al de N = 2; Cournot predice que el precio sigue bajando. 'cv' = coeficiente de variación de precios entre proveedores (asimetría de costos).
Fuente: elaboración propia con datos de OpenRouter (openrouter.ai) y LMArena.
