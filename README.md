<p align="center">
  <img src="https://latimpacto.org/wp-content/uploads/2023/11/Eafit.png" width="40%">
</p>

<h1 align="center">Analísis de Datos a Pruebas Universitarias SaberPro (2018-2021) Colombia</h1>

El ICFES (Instituto Colombiano para la Evaluación de la Calidad de la Educación) realiza anualmente las Pruebas Saber Pro para conocer el desarrollo de las competencias de los estudiantes que están por finalizar sus carreras Universitarias.

En este proyecto se plantea el uso de técnicas de ciencia de datos para analizar los resultados de las Pruebas Saber Pro de estudiantes entre 2018 y 2021, enriqueciendo los datos con el perfil socioeconómico proporcionado por el ICFES. El propósito es encontrar patrones y variables socioeconómicas que se relacionen significativamente con el rendimiento académico de los estudiantes y así contribuir con evidencia para el análisis y la toma de decisiones en el campo de la educación.

##  Estructura del repositorio

```
Auditoría/
│-- Prueba_Saber_Pro_Colombia.ipynb
│-- streamlit/
│   └── src/
│       ├── charts.py
│       ├── data.py
│       ├── eda.py
│       ├── groq_page.py
│       ├── kpis.py
│       ├── ui.py
│   ├── main_app.py
│   ├── mappings.py
│-- requirements.txt
│-- README.md

```

## Miembros del equipo

- Miguel Roldan Yepes
- Juan Camilo Cataño Zuleta
- Juan Manuel Agudelo Olarte
- Carlos José Muñoz Cabrera

## Set de datos

El set de datos que vamos a utilizar es [Saber Pro](https://drive.google.com/file/d/11I74e9DOLzR0_7XOV5Hr4ZaEsvjIbWR1/view?usp=sharing).
El dataset Saber Pro (2018-2021) incluye resultados de rendimiento, indicadores numéricos de la prueba y factores socioeconómicos/territoriales del alumnado (nivel socioeconómico, acceso a tecnología, educación recibida por los padres, programa y departamento).

## Streamlit

Para acceder a la aplicación de streamlit, hacer clic en [SaberPro](https://final-fundamentos-ciencia-datos.streamlit.app/)

## Créditos
RLX. UDEA/ai4eng 20252 - Pruebas Saber Pro Colombia. https://kaggle.com/competitions/udea-ai-4-eng-20252-pruebas-saber-pro-colombia, 2025. Kaggle.
