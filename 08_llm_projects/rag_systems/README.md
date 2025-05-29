# Sistema de Clasificación Documental Guiada (RAG)

## Propósito

Este sistema utiliza Recuperación Aumentada de Generación (RAG) para automatizar la clasificación de capturas de pantalla según lineamientos contenidos en un documento PDF guía. El sistema:

1. Extrae texto de imágenes mediante OCR
2. Procesa un PDF guía de referencia
3. Vectoriza el contenido para búsqueda semántica
4. Utiliza un modelo LLM local para clasificar y justificar resultados
5. Genera clasificaciones estructuradas con referencias al documento guía

Todo funciona 100% localmente para proteger la confidencialidad de los datos.

## Tech Stack

- **Lenguajes**: Python 3.9+
- **OCR**: DocTR (Document Text Recognition)
- **Procesamiento PDF**: PyMuPDF (fitz)
- **Embeddings**: SentenceTransformers
- **Vector Store**: Chroma DB
- **LLM Local**: Ollama + Mistral 7B/Phi-2
- **Orquestación**: LangChain/LlamaIndex
- **Testing**: Pytest

## Arquitectura

```
[Captura de pantalla] → [OCR] → [Texto extraído] → [Consulta RAG] → [LLM] → [Clasificaciones]
                                                   ↑
                         [PDF Guía] → [Parser] → [Vector DB]
```

## Setup

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar Ollama (para LLM local)
# Siga las instrucciones en https://ollama.ai/

# Descargar modelo
ollama pull mistral:7b
```

## Uso

```bash
# Ejecutar el clasificador con una imagen de entrada
python app/main.py --image data/samples/capture_01.png --guidelines data/guidelines/guide.pdf

# Modo interactivo
python app/main.py --interactive
```

## Estructura del Proyecto

```
document-guided-classifier/
├── README.md               # Esta documentación
├── requirements.txt        # Dependencias
├── setup.sh               # Script de instalación
├── data/
│   ├── guidelines/        # PDFs guía
│   └── samples/           # Capturas de pantalla de entrada
├── app/
│   ├── main.py            # Pipeline principal
│   ├── ocr.py             # Extracción OCR
│   ├── pdf_parser.py      # Procesamiento PDF
│   ├── vector_store.py    # Base de datos vectorial
│   ├── rag_engine.py      # Motor RAG con LLM
│   ├── prompts.py         # Templates de prompts
│   └── formatter.py       # Formateo de salidas
├── notebooks/             # Para experimentación
└── tests/                 # Tests unitarios
```

## Salida

El sistema genera un JSON estructurado con:
- Campo clasificado
- Valor sugerido
- Justificación textual
- Referencia al documento guía (página y texto)
- Comentario resumido para formulario

## Licencia

MIT
