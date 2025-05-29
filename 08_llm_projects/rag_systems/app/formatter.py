#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de formateo de salidas para el Sistema de Clasificación Documental Guiada.

Este módulo se encarga de procesar las respuestas del LLM y convertirlas en
formatos estructurados JSON para su uso en formularios o APIs.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Union

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Formateador de respuestas del LLM a estructuras JSON estandarizadas.
    
    Este formateador procesa las respuestas en texto del LLM y extrae
    información estructurada según el formato esperado.
    """
    
    def __init__(self):
        """Inicializa el formateador."""
        pass
    
    def extract_json_from_text(self, text: str) -> Optional[Dict]:
        """
        Extrae un objeto JSON de un texto que puede contener múltiples elementos.
        
        Args:
            text: Texto que puede contener JSON
            
        Returns:
            Optional[Dict]: Objeto JSON extraído o None si no se encuentra
        """
        # Buscar patrones de JSON entre comillas triples o backticks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, text)
        
        if matches:
            # Intentar parsear el primer match como JSON
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                logger.error(f"Error al decodificar JSON extraído: {matches[0][:100]}...")
        
        # Si no hay match con comillas/backticks, buscar en todo el texto
        try:
            # Intentar encontrar un objeto JSON en el texto completo
            json_pattern = r'(\{[\s\S]*\})'
            matches = re.search(json_pattern, text)
            
            if matches:
                return json.loads(matches.group(1))
        except (json.JSONDecodeError, AttributeError):
            logger.warning("No se pudo extraer JSON válido del texto")
        
        return None
    
    def format_classification_response(self, llm_response: str) -> Dict:
        """
        Formatea la respuesta de clasificación en una estructura consistente.
        
        Args:
            llm_response: Respuesta del LLM
            
        Returns:
            Dict: Clasificación estructurada
        """
        # Intentar extraer JSON directamente
        json_data = self.extract_json_from_text(llm_response)
        
        if json_data and "clasificaciones" in json_data:
            # Validar que el formato sea el esperado
            valid = all(
                isinstance(item, dict) and
                "campo" in item and
                "valor" in item and
                "justificación" in item and
                "referencia_guideline" in item and
                "comentario_formulario" in item
                for item in json_data["clasificaciones"]
            )
            
            if valid:
                return json_data
        
        # Si no hay JSON válido, realizar parsing manual
        return self.parse_classification_manually(llm_response)
    
    def parse_classification_manually(self, text: str) -> Dict:
        """
        Realiza parsing manual de la respuesta cuando no hay JSON explícito.
        
        Args:
            text: Texto de respuesta del LLM
            
        Returns:
            Dict: Clasificación estructurada
        """
        # Buscar patrones de clasificación en texto libre
        classifications = []
        
        # Patrones para identificar campos comunes
        field_patterns = [
            r'(?:Campo|campo|CAMPO):\s*([^\n]*)',
            r'(?:Valor|valor|VALOR):\s*([^\n]*)',
            r'(?:Justificaci[óo]n|justificaci[óo]n|JUSTIFICACI[ÓO]N):\s*([^\n]*)',
            r'(?:Referencia|referencia|REFERENCIA)[\w\s]*:\s*([^\n]*)',
            r'(?:P[áa]gina|p[áa]gina|P[ÁA]GINA):\s*(\d+)',
            r'(?:Comentario|comentario|COMENTARIO)[\w\s]*:\s*([^\n]*)'
        ]
        
        # Dividir por posibles delimitadores de secciones
        sections = re.split(r'\n\s*\n|\n---+\n|\n\*\*\*+\n|\n###\s', text)
        
        for section in sections:
            if not section.strip():
                continue
                
            # Detectar un posible campo de clasificación
            if re.search(field_patterns[0], section, re.IGNORECASE):
                try:
                    # Extraer componentes
                    campo_match = re.search(field_patterns[0], section, re.IGNORECASE)
                    valor_match = re.search(field_patterns[1], section, re.IGNORECASE)
                    justificacion_match = re.search(field_patterns[2], section, re.IGNORECASE)
                    referencia_match = re.search(field_patterns[3], section, re.IGNORECASE)
                    pagina_match = re.search(field_patterns[4], section, re.IGNORECASE)
                    comentario_match = re.search(field_patterns[5], section, re.IGNORECASE)
                    
                    # Si tenemos al menos campo y valor, crear una clasificación
                    if campo_match and valor_match:
                        campo = campo_match.group(1).strip()
                        valor = valor_match.group(1).strip()
                        justificacion = justificacion_match.group(1).strip() if justificacion_match else ""
                        referencia = referencia_match.group(1).strip() if referencia_match else ""
                        pagina = int(pagina_match.group(1)) if pagina_match else None
                        comentario = comentario_match.group(1).strip() if comentario_match else ""
                        
                        classifications.append({
                            "campo": campo,
                            "valor": valor,
                            "justificación": justificacion,
                            "referencia_guideline": {
                                "texto": referencia,
                                "página": pagina if pagina is not None else 0
                            },
                            "comentario_formulario": comentario or justificacion
                        })
                except Exception as e:
                    logger.error(f"Error al extraer clasificación de sección: {str(e)}")
        
        # Si no se encontraron clasificaciones, crear una genérica
        if not classifications:
            classifications.append({
                "campo": "Clasificación general",
                "valor": "No determinado",
                "justificación": "No se pudo extraer una clasificación estructurada de la respuesta",
                "referencia_guideline": {
                    "texto": "Sin referencia específica",
                    "página": 0
                },
                "comentario_formulario": "Se requiere revisión manual de la clasificación"
            })
            
            # Incluir la respuesta completa como contexto
            classifications[0]["respuesta_completa"] = text
        
        return {"clasificaciones": classifications}
    
    def format_chat_response(self, chat_response: Dict) -> Dict:
        """
        Formatea la respuesta de chat a un formato estructurado.
        
        Args:
            chat_response: Respuesta del sistema de chat
            
        Returns:
            Dict: Respuesta formateada
        """
        # Extraer y formatear referencias del documento
        references = []
        
        for chunk in chat_response.get("context_chunks", []):
            if "page" in chunk and chunk["page"]:
                references.append({
                    "texto": chunk.get("text", "").strip(),
                    "página": chunk.get("page", 0),
                    "sección": chunk.get("section", "")
                })
        
        return {
            "consulta": chat_response.get("query", ""),
            "respuesta": chat_response.get("response", ""),
            "referencias": references
        }


# Función auxiliar para uso directo
def format_llm_response(response: str, response_type: str = "classification") -> Dict:
    """
    Función auxiliar para formatear respuestas del LLM.
    
    Args:
        response: Respuesta del LLM
        response_type: Tipo de respuesta ("classification" o "chat")
        
    Returns:
        Dict: Respuesta formateada
    """
    formatter = ResponseFormatter()
    
    if response_type == "classification":
        return formatter.format_classification_response(response)
    elif response_type == "chat":
        # Para respuestas de chat, esperamos un diccionario
        if isinstance(response, str):
            return formatter.format_chat_response({"response": response})
        else:
            return formatter.format_chat_response(response)
    else:
        return {"error": f"Tipo de respuesta no soportado: {response_type}"}


if __name__ == "__main__":
    # Ejemplo de uso
    sample_response = """
    Basado en el análisis del texto y el documento guía, se presentan las siguientes clasificaciones:

    ```json
    {
      "clasificaciones": [
        {
          "campo": "Tipo de defecto",
          "valor": "Desalineación",
          "justificación": "La imagen muestra un desplazamiento visual entre elementos del mapa",
          "referencia_guideline": {
            "texto": "Los defectos de desalineación se identifican por el desplazamiento visual de elementos respecto a su referencia",
            "página": 42
          },
          "comentario_formulario": "Defecto de desalineación identificado según criterios de la página 42 del manual"
        },
        {
          "campo": "Prioridad",
          "valor": "Alta",
          "justificación": "La desalineación afecta significativamente la precisión de navegación",
          "referencia_guideline": {
            "texto": "Las desalineaciones que impactan la experiencia de navegación deben clasificarse como prioridad alta",
            "página": 86
          },
          "comentario_formulario": "Prioridad alta debido al impacto en la experiencia de navegación"
        }
      ]
    }
    ```
    """
    
    formatter = ResponseFormatter()
    result = formatter.format_classification_response(sample_response)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
