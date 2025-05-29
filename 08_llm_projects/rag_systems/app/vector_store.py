#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para la gestión de la base de datos vectorial.

Este módulo implementa la creación, gestión y consulta de una base de datos vectorial
para almacenar y recuperar embeddings de fragmentos de texto de forma semántica.
"""

import os
import logging
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from app.pdf_parser import PDFDocument, PDFChunk

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Base de datos vectorial para búsqueda semántica de documentos.
    
    Utiliza ChromaDB como backend y SentenceTransformers para generar embeddings.
    """
    
    def __init__(self, 
                 persist_dir: Optional[str] = None,
                 collection_name: str = "guidelines",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inicializa la base de datos vectorial.
        
        Args:
            persist_dir: Directorio para persistir la base de datos (None = en memoria)
            collection_name: Nombre de la colección de vectores
            embedding_model: Modelo para generar embeddings
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Inicializar cliente Chroma
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"Vector store persistente inicializada en: {persist_dir}")
        else:
            self.client = chromadb.Client()
            logger.info("Vector store en memoria inicializada")
            
        # Inicializar función de embeddings
        self._init_embedding_function()
        
        # Obtener o crear colección
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Colección '{collection_name}' preparada")
    
    def _init_embedding_function(self):
        """Inicializa la función de embeddings usando el modelo especificado."""
        try:
            # Usar SentenceTransformers 
            self.model = SentenceTransformer(self.embedding_model_name)
            
            # Crear función de embedding personalizada
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            
            logger.info(f"Modelo de embeddings inicializado: {self.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"Error al inicializar modelo de embeddings: {str(e)}")
            raise
    
    def add_pdf_document(self, document: PDFDocument, batch_size: int = 32) -> int:
        """
        Añade un documento PDF a la base de datos vectorial.
        
        Args:
            document: Documento PDF procesado
            batch_size: Tamaño de lote para indexación
            
        Returns:
            int: Número de chunks indexados
        """
        # Extraer chunks y metadatos
        chunks = document.chunks
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            logger.warning("No hay chunks para indexar en el documento")
            return 0
        
        logger.info(f"Indexando {total_chunks} chunks del documento: {document.title}")
        
        # Procesar chunks en lotes
        for i in tqdm(range(0, total_chunks, batch_size), desc="Indexando lotes"):
            batch_chunks = chunks[i:i+batch_size]
            
            # Preparar datos para indexación
            ids = [chunk.chunk_id for chunk in batch_chunks]
            texts = [chunk.text for chunk in batch_chunks]
            metadatas = []
            
            for chunk in batch_chunks:
                metadatas.append({
                    "page": chunk.page_number,
                    "section": chunk.section,
                    "subsection": chunk.subsection,
                    "document_title": document.title,
                    "document_path": document.path,
                })
            
            # Añadir a la colección
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        
        logger.info(f"Indexación completada: {total_chunks} chunks añadidos a la colección")
        return total_chunks
    
    def add_texts(self, 
                  texts: List[str], 
                  metadatas: Optional[List[Dict]] = None, 
                  ids: Optional[List[str]] = None) -> List[str]:
        """
        Añade textos genéricos a la base de datos.
        
        Args:
            texts: Lista de textos a añadir
            metadatas: Lista de metadatos para cada texto
            ids: IDs para cada texto
            
        Returns:
            List[str]: IDs de los documentos añadidos
        """
        if not ids:
            ids = [f"doc_{i}" for i in range(len(texts))]
            
        if not metadatas:
            metadatas = [{} for _ in range(len(texts))]
            
        # Añadir a la colección
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        return ids
    
    def search(self, 
               query: str, 
               n_results: int = 5, 
               filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Realiza una búsqueda semántica en la base de datos.
        
        Args:
            query: Texto de consulta
            n_results: Número de resultados a retornar
            filter_dict: Filtros para la búsqueda (por metadatos)
            
        Returns:
            List[Dict]: Resultados de la búsqueda
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        
        # Procesar resultados para formato más amigable
        processed_results = []
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                processed_results.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'][0] else {},
                    "id": results['ids'][0][i] if results['ids'][0] else None,
                    "score": results['distances'][0][i] if results['distances'][0] else None
                })
        
        return processed_results
    
    def delete_collection(self):
        """Elimina la colección de la base de datos."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Colección '{self.collection_name}' eliminada")
        
    def get_collection_stats(self) -> Dict:
        """
        Obtiene estadísticas de la colección actual.
        
        Returns:
            Dict: Estadísticas de la colección
        """
        try:
            count = self.collection.count()
            
            # Ejemplo para obtener todos los metadatos (usar con precaución en colecciones grandes)
            if count < 1000:  # Solo para colecciones pequeñas
                all_metadatas = self.collection.get()["metadatas"]
                
                # Contar documentos por página
                pages = {}
                sections = set()
                
                for meta in all_metadatas:
                    page = meta.get("page", None)
                    if page is not None:
                        pages[page] = pages.get(page, 0) + 1
                        
                    section = meta.get("section", "")
                    if section:
                        sections.add(section)
                
                return {
                    "count": count,
                    "pages": pages,
                    "unique_sections": len(sections),
                    "collection_name": self.collection_name
                }
            else:
                return {"count": count, "collection_name": self.collection_name}
                
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {str(e)}")
            return {"error": str(e)}


# Función auxiliar para crear y usar vector store
def create_vector_store_from_pdf(
    pdf_path: Union[str, Path],
    persist_dir: Optional[Union[str, Path]] = None,
    collection_name: Optional[str] = None,
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> Tuple[VectorStore, PDFDocument]:
    """
    Función auxiliar para crear una base de datos vectorial a partir de un PDF.
    
    Args:
        pdf_path: Ruta al archivo PDF
        persist_dir: Directorio para persistir los embeddings
        collection_name: Nombre de la colección (por defecto basado en nombre del archivo)
        embedding_model: Modelo para embeddings
        
    Returns:
        Tuple[VectorStore, PDFDocument]: Vector store y documento procesado
    """
    from app.pdf_parser import PDFParser
    
    # Procesar el PDF
    parser = PDFParser()
    document = parser.parse_pdf(pdf_path)
    
    # Determinar nombre de colección
    if not collection_name:
        collection_name = Path(pdf_path).stem.lower().replace(" ", "_")
    
    # Crear vector store
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
    
    vector_store = VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    
    # Añadir documento
    vector_store.add_pdf_document(document)
    
    return vector_store, document


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        persist_dir = sys.argv[2] if len(sys.argv) > 2 else "vector_store"
        
        vector_store, document = create_vector_store_from_pdf(pdf_file, persist_dir)
        
        print(f"Base de datos vectorial creada para: {document.title}")
        print(f"Total de chunks: {len(document.chunks)}")
        
        # Ejemplo de búsqueda
        query = "¿Cómo clasificar un caso de desalineación?"
        results = vector_store.search(query, n_results=3)
        
        print(f"\nResultados para '{query}':")
        for i, result in enumerate(results):
            print(f"\n--- Resultado {i+1} (Página {result['metadata'].get('page')}) ---")
            print(f"Puntuación: {result['score']:.4f}")
            print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
    else:
        print("Uso: python vector_store.py <ruta_pdf> [directorio_persistencia]")
