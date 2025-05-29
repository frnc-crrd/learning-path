#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motor RAG para el Sistema de Clasificación Documental Guiada.

Este módulo implementa el pipeline de Recuperación Aumentada de Generación (RAG)
que integra el OCR, vectorización y modelo LLM para generar clasificaciones.
"""

import os
import json
import logging
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import requests

from app.ocr import OCREngine
from app.vector_store import VectorStore
from app.prompts import (
    SYSTEM_TEMPLATE,
    get_rag_query_prompt,
    get_structured_classification_prompt,
    get_chat_query_prompt
)

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMProvider:
    """
    Proveedor de modelo de lenguaje para clasificación.
    
    Esta clase abstrae la interacción con el LLM, soportando
    diferentes backends como Ollama o API directas.
    """
    
    def __init__(self, 
                 provider: str = "ollama",
                 model_name: str = "mistral:7b",
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 2048):
        """
        Inicializa el proveedor LLM.
        
        Args:
            provider: Proveedor LLM ("ollama" o "api")
            model_name: Nombre del modelo a utilizar
            api_base: URL base para API (si se usa proveedor "api")
            api_key: Clave de API (si se requiere)
            temperature: Temperatura para generación (0.0-1.0)
            max_tokens: Máximo de tokens a generar
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Inicializando LLM Provider: {provider}, modelo: {model_name}")
        
        # Verificar disponibilidad del proveedor
        self._check_provider()
    
    def _check_provider(self):
        """Verifica disponibilidad del proveedor LLM."""
        if self.provider == "ollama":
            try:
                # Verificar si Ollama está disponible
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                if result.returncode != 0:
                    logger.warning(f"Ollama no está disponible: {result.stderr}")
                    raise RuntimeError(f"Ollama no está disponible: {result.stderr}")
                
                # Verificar si el modelo está disponible
                if self.model_name not in result.stdout:
                    logger.warning(f"Modelo '{self.model_name}' no disponible en Ollama")
                    logger.info(f"Modelos disponibles: {result.stdout}")
                    
                    # Opcionalmente descomentar para descargar automáticamente
                    # logger.info(f"Descargando modelo {self.model_name}...")
                    # subprocess.run(["ollama", "pull", self.model_name], check=True)
            except Exception as e:
                logger.error(f"Error al verificar Ollama: {str(e)}")
                raise
        elif self.provider == "api":
            if not self.api_base:
                raise ValueError("Se requiere api_base para el proveedor API")
    
    def generate(self, 
                 system_prompt: str,
                 user_prompt: str, 
                 stream: bool = False) -> str:
        """
        Genera texto usando el LLM.
        
        Args:
            system_prompt: Prompt de sistema
            user_prompt: Prompt de usuario 
            stream: Si debe transmitir la respuesta por partes
            
        Returns:
            str: Texto generado
        """
        if self.provider == "ollama":
            return self._generate_ollama(system_prompt, user_prompt, stream)
        elif self.provider == "api":
            return self._generate_api(system_prompt, user_prompt, stream)
        else:
            raise ValueError(f"Proveedor no soportado: {self.provider}")
    
    def _generate_ollama(self, system_prompt: str, user_prompt: str, stream: bool) -> str:
        """Genera texto usando Ollama."""
        try:
            # URL de la API local de Ollama
            url = "http://localhost:11434/api/generate"
            
            # Preparar payload
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": stream,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            # Realizar petición
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # Procesar respuesta
            if stream:
                # Manejar streaming
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        full_response += chunk.get('response', '')
                        
                        # Verificar si es el final
                        if chunk.get('done', False):
                            break
                
                return full_response
            else:
                # Respuesta completa
                return response.json().get('response', '')
            
        except Exception as e:
            logger.error(f"Error en generación con Ollama: {str(e)}")
            # Manejo de errores específicos
            if isinstance(e, requests.exceptions.ConnectionError):
                return "Error: No se pudo conectar con Ollama. ¿Está ejecutándose el servicio?"
            return f"Error al generar respuesta: {str(e)}"
    
    def _generate_api(self, system_prompt: str, user_prompt: str, stream: bool) -> str:
        """Genera texto usando una API externa."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            
            # Preparar payload según formato común de API
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": stream
            }
            
            # Realizar petición
            response = requests.post(self.api_base, headers=headers, json=payload)
            response.raise_for_status()
            
            # Procesar respuesta
            if stream:
                # Manejar streaming
                full_response = ""
                for line in response.iter_lines():
                    if line and line.strip() != b'data: [DONE]':
                        try:
                            # Remover 'data: ' al inicio si existe
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                line_str = line_str[6:]
                            
                            chunk = json.loads(line_str)
                            content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            full_response += content
                        except:
                            pass
                
                return full_response
            else:
                # Respuesta completa
                data = response.json()
                return data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
        except Exception as e:
            logger.error(f"Error en generación con API: {str(e)}")
            return f"Error al generar respuesta: {str(e)}"


class RAGEngine:
    """
    Motor RAG para clasificación basada en documentos guía.
    
    Implementa el pipeline completo de recuperación y generación
    para clasificar imágenes según documentos guía.
    """
    
    def __init__(self,
                 vector_store: VectorStore,
                 llm_provider: LLMProvider,
                 ocr_engine: Optional[OCREngine] = None,
                 ocr_engine_name: str = "doctr"):
        """
        Inicializa el motor RAG.
        
        Args:
            vector_store: Base de datos vectorial
            llm_provider: Proveedor de LLM
            ocr_engine: Motor OCR preconfigurado (opcional)
            ocr_engine_name: Nombre del motor OCR si se debe crear
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        
        # Inicializar OCR si no se proporciona
        if ocr_engine is None:
            self.ocr_engine = OCREngine(engine=ocr_engine_name)
        else:
            self.ocr_engine = ocr_engine
    
    def process_image(self, 
                      image_path: Union[str, Path], 
                      fields_to_classify: Optional[List[str]] = None,
                      n_results: int = 5) -> Dict:
        """
        Procesa una imagen para clasificación.
        
        Args:
            image_path: Ruta de la imagen a procesar
            fields_to_classify: Lista de campos específicos a clasificar (opcional)
            n_results: Número de resultados a recuperar de la base vectorial
            
        Returns:
            Dict: Resultados de la clasificación
        """
        # 1. Extraer texto de la imagen con OCR
        logger.info(f"Extrayendo texto de la imagen: {image_path}")
        ocr_result = self.ocr_engine.process_image(image_path)
        extracted_text = ocr_result["text"]
        
        if not extracted_text.strip():
            logger.warning("No se pudo extraer texto de la imagen")
            return {
                "error": "No se pudo extraer texto de la imagen",
                "ocr_confidence": ocr_result.get("confidence", 0)
            }
        
        # 2. Buscar contexto relevante en la base vectorial
        logger.info("Buscando contexto en la base vectorial")
        context_chunks = self.vector_store.search(extracted_text, n_results=n_results)
        
        if not context_chunks:
            logger.warning("No se encontraron fragmentos relevantes en la base vectorial")
            return {
                "error": "No se encontraron fragmentos relevantes en la guía",
                "extracted_text": extracted_text
            }
        
        # 3. Generar prompt según necesidad
        if fields_to_classify:
            # Clasificación estructurada de campos específicos
            prompt = get_structured_classification_prompt(
                extracted_text=extracted_text,
                context_chunks=context_chunks,
                fields_to_classify=fields_to_classify
            )
        else:
            # Clasificación general
            prompt = get_rag_query_prompt(
                extracted_text=extracted_text,
                context_chunks=context_chunks
            )
        
        # 4. Generar clasificación con LLM
        logger.info("Generando clasificación con LLM")
        llm_response = self.llm_provider.generate(
            system_prompt=SYSTEM_TEMPLATE,
            user_prompt=prompt
        )
        
        # 5. Formatear y devolver resultados
        return {
            "extracted_text": extracted_text,
            "ocr_confidence": ocr_result.get("confidence", 0),
            "classification": llm_response,
            "context_chunks": [
                {
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "page": chunk.get("metadata", {}).get("page", ""),
                    "section": chunk.get("metadata", {}).get("section", ""),
                    "score": chunk.get("score", 0)
                }
                for chunk in context_chunks
            ]
        }
    
    def chat_with_document(self, query: str, n_results: int = 3) -> Dict:
        """
        Permite chatear con el documento guía.
        
        Args:
            query: Consulta del usuario
            n_results: Número de fragmentos a recuperar
            
        Returns:
            Dict: Respuesta del sistema
        """
        # 1. Buscar contexto relevante para la consulta
        logger.info(f"Buscando contexto para consulta: {query}")
        context_chunks = self.vector_store.search(query, n_results=n_results)
        
        if not context_chunks:
            logger.warning("No se encontraron fragmentos relevantes para la consulta")
            return {
                "error": "No se encontró información relacionada en la guía",
                "query": query
            }
        
        # 2. Generar prompt para chat
        prompt = get_chat_query_prompt(
            user_query=query,
            context_chunks=context_chunks
        )
        
        # 3. Generar respuesta con LLM
        logger.info("Generando respuesta con LLM")
        llm_response = self.llm_provider.generate(
            system_prompt=SYSTEM_TEMPLATE,
            user_prompt=prompt
        )
        
        # 4. Devolver respuesta con contexto
        return {
            "query": query,
            "response": llm_response,
            "context_chunks": [
                {
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "page": chunk.get("metadata", {}).get("page", ""),
                    "section": chunk.get("metadata", {}).get("section", "")
                }
                for chunk in context_chunks
            ]
        }


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    from app.vector_store import create_vector_store_from_pdf
    
    if len(sys.argv) > 2:
        pdf_file = sys.argv[1]
        image_file = sys.argv[2]
        
        # Crear vector store
        vector_store, _ = create_vector_store_from_pdf(pdf_file)
        
        # Crear LLM provider
        llm = LLMProvider(provider="ollama", model_name="mistral:7b")
        
        # Crear motor RAG
        rag_engine = RAGEngine(vector_store=vector_store, llm_provider=llm)
        
        # Procesar imagen
        result = rag_engine.process_image(
            image_path=image_file,
            fields_to_classify=["Tipo de caso", "Prioridad", "Área responsable"]
        )
        
        # Mostrar resultados
        print("\n=== Texto extraído ===")
        print(result["extracted_text"][:300] + "..." if len(result["extracted_text"]) > 300 else result["extracted_text"])
        
        print("\n=== Clasificación ===")
        print(result["classification"])
        
    else:
        print("Uso: python rag_engine.py <pdf_guía> <imagen>")
