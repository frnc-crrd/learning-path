#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de plantillas de prompts para interacción con LLM.

Este módulo contiene los templates de prompts optimizados para interacción con el
modelo de lenguaje en diferentes escenarios del sistema.
"""

from typing import Dict, List, Optional, Any, Union
from string import Template

# Sistema base: define el rol y contexto del asistente
SYSTEM_TEMPLATE = """Eres un asistente experto en clasificación de casos según guías oficiales.
Tu tarea es analizar el texto extraído de una captura de pantalla y determinar la clasificación correcta
basándote en los lineamientos oficiales del documento guía.

Debes proporcionar clasificaciones precisas y bien fundamentadas, respaldadas por fragmentos exactos
del documento guía. Incluye siempre la página y sección específica que respalda tu decisión.

Tu respuesta debe ser detallada, bien estructurada y objetiva, siguiendo estrictamente las reglas del
documento guía proporcionado.
"""

# Para procesar nueva consulta con contexto extraído de la guía
RAG_QUERY_TEMPLATE = Template("""
# Texto extraído de la captura de pantalla
$extracted_text

# Fragmentos relevantes del documento guía
$context_chunks

# Instrucciones
Analiza el texto extraído de la captura de pantalla y clasifícalo según los fragmentos del documento guía.
Para cada campo de clasificación que identifiques, debes proporcionar:

1. Nombre del campo a clasificar
2. Valor sugerido para el campo
3. Justificación de la clasificación basada en el texto de la captura
4. Referencia textual exacta del documento guía que sustenta tu decisión
5. Número de página donde se encuentra esta referencia
6. Un comentario conciso pero completo para incluir en el formulario de clasificación

Tu análisis debe ser estructurado, organizado por campos de clasificación, y basado exclusivamente en
la evidencia proporcionada por el documento guía.
""")

# Para clasificación estructurada
STRUCTURED_CLASSIFICATION_TEMPLATE = Template("""
# Texto extraído de la captura de pantalla
$extracted_text

# Fragmentos relevantes del documento guía
$context_chunks

# Campos específicos a clasificar
$fields_to_classify

# Instrucciones
Analiza el texto extraído y proporciona una clasificación estructurada según los campos solicitados.
Utiliza exclusivamente la información de los fragmentos del documento guía como referencia.

Para cada uno de los campos solicitados, debes proporcionar exactamente:
1. El valor recomendado para el campo
2. Una justificación precisa basada en el texto analizado
3. La cita textual del documento guía que fundamenta la decisión
4. El número de página y sección donde aparece esta referencia
5. Un comentario conciso para el formulario que resuma la clasificación

Estructura tu respuesta como un objeto JSON válido, siguiendo este formato exacto:
```json
{
  "clasificaciones": [
    {
      "campo": "[Nombre del campo]",
      "valor": "[Valor seleccionado]",
      "justificación": "[Explicación de la clasificación]",
      "referencia_guideline": {
        "texto": "[Cita textual del documento]",
        "página": número_de_página
      },
      "comentario_formulario": "[Resumen conciso para el formulario]"
    },
    // Más clasificaciones...
  ]
}
```
""")

# Para conversación con el sistema sobre el contenido del documento guía
CHAT_QUERY_TEMPLATE = Template("""
# Contexto: Fragmentos relevantes del documento guía
$context_chunks

# Consulta del usuario
$user_query

# Instrucciones
Responde a la consulta del usuario utilizando exclusivamente la información proporcionada 
en los fragmentos del documento guía. Tu respuesta debe:

1. Ser precisa y basada únicamente en los fragmentos proporcionados
2. Citar las partes específicas del documento guía que respaldan tu respuesta
3. Mencionar las páginas de referencia para cada información proporcionada
4. Ser clara y directa, respondiendo exactamente lo que se pregunta
5. Admitir honestamente cuando la información no esté contenida en los fragmentos

No inventes información ni agregues conocimiento que no esté presente en los fragmentos proporcionados.
""")

# Para explicación detallada de una clasificación específica
EXPLAIN_CLASSIFICATION_TEMPLATE = Template("""
# Texto extraído de la captura de pantalla
$extracted_text

# Campo específico
$specific_field

# Clasificación actual
$current_classification

# Fragmentos relevantes del documento guía
$context_chunks

# Instrucciones
Explica detalladamente por qué se asignó el valor "$current_classification" al campo "$specific_field"
basándote en el texto extraído y el documento guía.

Tu explicación debe:
1. Identificar elementos específicos en el texto extraído que conducen a esta clasificación
2. Citar fragmentos precisos del documento guía que sustentan esta decisión
3. Explicar el proceso de razonamiento paso a paso
4. Mencionar las páginas específicas del documento guía que respaldan esta clasificación
5. Evaluar el nivel de confianza en esta clasificación (alto, medio, bajo)

Estructura tu respuesta para que sea clara y educativa, justificando completamente la clasificación.
""")

# Para generar resumen ejecutivo de una clasificación completa
SUMMARY_TEMPLATE = Template("""
# Clasificaciones realizadas
$classifications_json

# Instrucciones
Genera un resumen ejecutivo de las clasificaciones realizadas. Este resumen debe:

1. Ser conciso y profesional, adecuado para un informe formal
2. Destacar los campos más importantes y sus clasificaciones
3. Mencionar las principales referencias del documento guía utilizadas
4. Presentar un nivel general de confianza en las clasificaciones
5. Tener un máximo de 150 palabras

El resumen debe ser informativo y estructurado para que un supervisor pueda entender 
rápidamente el caso y las decisiones tomadas.
""")


def get_rag_query_prompt(extracted_text: str, context_chunks: List[Dict]) -> str:
    """
    Genera un prompt para consulta RAG con texto extraído y contexto.
    
    Args:
        extracted_text: Texto extraído de la imagen
        context_chunks: Fragmentos relevantes del documento guía
        
    Returns:
        str: Prompt completo para el modelo
    """
    # Formatear chunks de contexto
    formatted_chunks = []
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text", "")
        page = chunk.get("metadata", {}).get("page", "")
        section = chunk.get("metadata", {}).get("section", "")
        
        formatted_chunks.append(f"--- Fragmento {i+1} (Página {page}) ---")
        if section:
            formatted_chunks.append(f"Sección: {section}")
        formatted_chunks.append(text)
        formatted_chunks.append("")
    
    context_text = "\n".join(formatted_chunks)
    
    return RAG_QUERY_TEMPLATE.substitute(
        extracted_text=extracted_text,
        context_chunks=context_text
    )


def get_structured_classification_prompt(
    extracted_text: str, 
    context_chunks: List[Dict],
    fields_to_classify: List[str]
) -> str:
    """
    Genera un prompt para clasificación estructurada.
    
    Args:
        extracted_text: Texto extraído de la imagen
        context_chunks: Fragmentos relevantes del documento guía
        fields_to_classify: Lista de campos a clasificar
        
    Returns:
        str: Prompt completo para el modelo
    """
    # Formatear chunks de contexto
    formatted_chunks = []
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text", "")
        page = chunk.get("metadata", {}).get("page", "")
        section = chunk.get("metadata", {}).get("section", "")
        
        formatted_chunks.append(f"--- Fragmento {i+1} (Página {page}) ---")
        if section:
            formatted_chunks.append(f"Sección: {section}")
        formatted_chunks.append(text)
        formatted_chunks.append("")
    
    context_text = "\n".join(formatted_chunks)
    fields_text = "- " + "\n- ".join(fields_to_classify)
    
    return STRUCTURED_CLASSIFICATION_TEMPLATE.substitute(
        extracted_text=extracted_text,
        context_chunks=context_text,
        fields_to_classify=fields_text
    )


def get_chat_query_prompt(user_query: str, context_chunks: List[Dict]) -> str:
    """
    Genera un prompt para conversación con el sistema.
    
    Args:
        user_query: Consulta del usuario
        context_chunks: Fragmentos relevantes del documento guía
        
    Returns:
        str: Prompt completo para el modelo
    """
    # Formatear chunks de contexto
    formatted_chunks = []
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text", "")
        page = chunk.get("metadata", {}).get("page", "")
        section = chunk.get("metadata", {}).get("section", "")
        
        formatted_chunks.append(f"--- Fragmento {i+1} (Página {page}) ---")
        if section:
            formatted_chunks.append(f"Sección: {section}")
        formatted_chunks.append(text)
        formatted_chunks.append("")
    
    context_text = "\n".join(formatted_chunks)
    
    return CHAT_QUERY_TEMPLATE.substitute(
        context_chunks=context_text,
        user_query=user_query
    )


if __name__ == "__main__":
    # Ejemplo para probar los templates
    extracted_text = "Ejemplo de texto extraído de una captura de pantalla."
    context_chunks = [
        {"text": "Texto de ejemplo del documento guía.", "metadata": {"page": 12, "section": "Clasificación"}}
    ]
    fields = ["Tipo de caso", "Prioridad", "Departamento responsable"]
    
    print("\n=== Ejemplo de RAG Query ===")
    print(get_rag_query_prompt(extracted_text, context_chunks))
    
    print("\n=== Ejemplo de Clasificación Estructurada ===")
    print(get_structured_classification_prompt(extracted_text, context_chunks, fields))
