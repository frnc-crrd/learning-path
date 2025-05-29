#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo principal para el Sistema de Clasificación Documental Guiada.

Este módulo implementa la interfaz de línea de comandos y orquesta
el flujo completo del sistema RAG para la clasificación de imágenes.
"""

import os
import argparse
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from app.ocr import OCREngine
from app.pdf_parser import PDFParser, extract_pdf_content, PDFDocument
from app.vector_store import VectorStore, create_vector_store_from_pdf
from app.rag_engine import RAGEngine, LLMProvider
from app.formatter import ResponseFormatter

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("classifier.log")
    ]
)
logger = logging.getLogger(__name__)

# Consola para salida enriquecida
console = Console()


class DocumentGuidedClassifier:
    """
    Clasificador guiado por documentos basado en RAG.
    
    Esta clase orquesta todo el flujo de trabajo del sistema:
    1. Procesar PDF guía
    2. Extraer texto de imágenes
    3. Consultar base de datos vectorial
    4. Generar clasificaciones con LLM
    5. Formatear resultados
    """
    
    def __init__(self, 
                 guidelines_path: Optional[str] = None,
                 vector_store_dir: str = "vector_store",
                 ocr_engine: str = "doctr",
                 llm_model: str = "mistral:7b"):
        """
        Inicializa el clasificador.
        
        Args:
            guidelines_path: Ruta al PDF guía (opcional al inicio)
            vector_store_dir: Directorio para base de datos vectorial
            ocr_engine: Motor OCR a utilizar
            llm_model: Modelo LLM a utilizar
        """
        self.guidelines_path = guidelines_path
        self.vector_store_dir = vector_store_dir
        self.ocr_engine_name = ocr_engine
        self.llm_model_name = llm_model
        
        # Componentes del sistema (inicializados bajo demanda)
        self.ocr_engine = None
        self.vector_store = None
        self.llm_provider = None
        self.rag_engine = None
        self.formatter = None
        self.pdf_document = None
        
        # Inicializar componentes básicos
        self._init_components()
    
    def _init_components(self):
        """Inicializa componentes básicos del sistema."""
        # OCR
        self.ocr_engine = OCREngine(engine=self.ocr_engine_name)
        
        # Formateador de respuestas
        self.formatter = ResponseFormatter()
        
        # LLM Provider
        self.llm_provider = LLMProvider(
            provider="ollama", 
            model_name=self.llm_model_name,
            temperature=0.1
        )
        
        logger.info("Componentes básicos inicializados")
    
    def setup_vector_store(self, 
                          guidelines_path: Optional[str] = None,
                          force_rebuild: bool = False):
        """
        Configura la base de datos vectorial.
        
        Args:
            guidelines_path: Ruta al PDF guía (si no se especificó antes)
            force_rebuild: Si debe reconstruir la base de datos
        """
        # Actualizar ruta de guía si se proporciona
        if guidelines_path:
            self.guidelines_path = guidelines_path
        
        if not self.guidelines_path:
            raise ValueError("Se requiere una ruta al PDF guía")
        
        # Verificar si la base de datos ya existe
        collection_name = Path(self.guidelines_path).stem.lower().replace(" ", "_")
        vector_store_path = Path(self.vector_store_dir)
        
        if vector_store_path.exists() and not force_rebuild:
            logger.info(f"Cargando base de datos vectorial existente: {self.vector_store_dir}")
            self.vector_store = VectorStore(
                persist_dir=self.vector_store_dir,
                collection_name=collection_name
            )
        else:
            logger.info(f"Creando nueva base de datos vectorial para: {self.guidelines_path}")
            
            # Procesar PDF y crear vector store
            with console.status("[bold green]Procesando PDF guía..."):
                self.vector_store, self.pdf_document = create_vector_store_from_pdf(
                    pdf_path=self.guidelines_path,
                    persist_dir=self.vector_store_dir,
                    collection_name=collection_name
                )
        
        # Inicializar RAG Engine
        self.rag_engine = RAGEngine(
            vector_store=self.vector_store,
            llm_provider=self.llm_provider,
            ocr_engine=self.ocr_engine
        )
        
        logger.info("Sistema RAG completamente inicializado")
    
    def process_image(self, 
                     image_path: str,
                     fields_to_classify: Optional[List[str]] = None,
                     output_path: Optional[str] = None) -> Dict:
        """
        Procesa una imagen y genera clasificaciones.
        
        Args:
            image_path: Ruta a la imagen a procesar
            fields_to_classify: Campos específicos a clasificar
            output_path: Ruta para guardar resultados
            
        Returns:
            Dict: Resultados de la clasificación
        """
        if not self.rag_engine:
            raise RuntimeError("El sistema RAG no está inicializado. Ejecute setup_vector_store primero.")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        logger.info(f"Procesando imagen: {image_path}")
        
        # Procesar imagen con RAG
        with console.status("[bold green]Procesando imagen con OCR y RAG..."):
            result = self.rag_engine.process_image(
                image_path=image_path,
                fields_to_classify=fields_to_classify
            )
        
        # Formatear resultados
        formatted_result = self.formatter.format_classification_response(result["classification"])
        
        # Añadir metadatos
        result_with_metadata = {
            "metadata": {
                "imagen": Path(image_path).name,
                "guia": Path(self.guidelines_path).name,
                "ocr_confidence": result.get("ocr_confidence", 0),
            },
            **formatted_result
        }
        
        # Guardar resultados si se especifica ruta
        if output_path:
            output_dir = Path(output_path).parent
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_with_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Resultados guardados en: {output_path}")
        
        return result_with_metadata
    
    def chat_with_document(self, query: str) -> Dict:
        """
        Permite chatear con el documento guía.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Dict: Respuesta del sistema
        """
        if not self.rag_engine:
            raise RuntimeError("El sistema RAG no está inicializado. Ejecute setup_vector_store primero.")
        
        logger.info(f"Procesando consulta: {query}")
        
        # Consultar RAG
        with console.status("[bold green]Consultando documento guía..."):
            result = self.rag_engine.chat_with_document(query)
        
        # Formatear respuesta
        return self.formatter.format_chat_response(result)
    
    def run_interactive(self):
        """Ejecuta una sesión interactiva de chat con el documento guía."""
        if not self.rag_engine:
            raise RuntimeError("El sistema RAG no está inicializado. Ejecute setup_vector_store primero.")
        
        console.print(Panel.fit(
            "[bold green]Sesión interactiva con el documento guía[/bold green]\n"
            "Escriba sus consultas o '/salir' para finalizar. '/imagen [ruta]' para procesar imagen."
        ))
        
        while True:
            try:
                query = console.input("\n[bold blue]Consulta:[/bold blue] ")
                
                if not query.strip():
                    continue
                    
                if query.lower() in ("/salir", "/exit", "/quit", "q"):
                    break
                
                if query.lower().startswith("/imagen ") or query.lower().startswith("/image "):
                    # Procesar imagen
                    parts = query.split(" ", 1)
                    if len(parts) < 2 or not parts[1].strip():
                        console.print("[bold red]Especifique la ruta de la imagen[/bold red]")
                        continue
                    
                    image_path = parts[1].strip()
                    try:
                        result = self.process_image(image_path)
                        self._display_classification(result)
                    except Exception as e:
                        console.print(f"[bold red]Error al procesar imagen:[/bold red] {str(e)}")
                else:
                    # Consulta normal
                    try:
                        result = self.chat_with_document(query)
                        self._display_chat_response(result)
                    except Exception as e:
                        console.print(f"[bold red]Error en la consulta:[/bold red] {str(e)}")
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Sesión finalizada por el usuario[/yellow]")
                break
            except Exception as e:
                console.print(f"[bold red]Error inesperado:[/bold red] {str(e)}")
    
    def _display_classification(self, result: Dict):
        """Muestra clasificación en consola con formato enriquecido."""
        console.print("\n[bold green]Resultados de la clasificación:[/bold green]")
        
        for i, item in enumerate(result.get("clasificaciones", [])):
            console.print(Panel.fit(
                f"[bold cyan]Campo:[/bold cyan] {item.get('campo')}\n"
                f"[bold cyan]Valor:[/bold cyan] {item.get('valor')}\n"
                f"[bold cyan]Justificación:[/bold cyan] {item.get('justificación')}\n"
                f"[bold cyan]Referencia (p. {item.get('referencia_guideline', {}).get('página', 'N/A')}):[/bold cyan] "
                f"{item.get('referencia_guideline', {}).get('texto', '')}\n"
                f"[bold cyan]Comentario:[/bold cyan] {item.get('comentario_formulario')}",
                title=f"Clasificación {i+1}"
            ))
    
    def _display_chat_response(self, result: Dict):
        """Muestra respuesta de chat en consola con formato enriquecido."""
        console.print("\n[bold green]Respuesta:[/bold green]")
        console.print(Markdown(result.get("respuesta", "")))
        
        if result.get("referencias"):
            table = Table(title="Referencias del documento guía")
            table.add_column("Página", style="cyan")
            table.add_column("Sección", style="green")
            table.add_column("Texto", style="white")
            
            for ref in result.get("referencias"):
                table.add_row(
                    str(ref.get("página", "N/A")),
                    ref.get("sección", ""),
                    ref.get("texto", "")[:100] + "..." if len(ref.get("texto", "")) > 100 else ref.get("texto", "")
                )
            
            console.print(table)


def main():
    """Función principal para ejecución desde línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Sistema de Clasificación Documental Guiada con RAG"
    )
    
    parser.add_argument(
        "--guidelines", "-g", type=str, help="Ruta al archivo PDF guía"
    )
    parser.add_argument(
        "--image", "-i", type=str, help="Ruta a la imagen para clasificar"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Ruta para guardar resultados JSON"
    )
    parser.add_argument(
        "--vector-store", type=str, default="vector_store", 
        help="Directorio para base de datos vectorial"
    )
    parser.add_argument(
        "--fields", "-f", type=str, nargs="+", 
        help="Campos específicos a clasificar"
    )
    parser.add_argument(
        "--ocr-engine", type=str, default="doctr", choices=["doctr", "easyocr"],
        help="Motor OCR a utilizar"
    )
    parser.add_argument(
        "--llm-model", type=str, default="mistral:7b",
        help="Modelo LLM a utilizar"
    )
    parser.add_argument(
        "--rebuild-db", action="store_true",
        help="Reconstruir la base de datos vectorial"
    )
    parser.add_argument(
        "--interactive", "-it", action="store_true",
        help="Iniciar sesión interactiva"
    )
    
    args = parser.parse_args()
    
    try:
        # Crear clasificador
        classifier = DocumentGuidedClassifier(
            guidelines_path=args.guidelines,
            vector_store_dir=args.vector_store,
            ocr_engine=args.ocr_engine,
            llm_model=args.llm_model
        )
        
        # Configurar vector store
        classifier.setup_vector_store(
            guidelines_path=args.guidelines,
            force_rebuild=args.rebuild_db
        )
        
        if args.interactive:
            # Modo interactivo
            classifier.run_interactive()
        elif args.image:
            # Procesar imagen
            result = classifier.process_image(
                image_path=args.image,
                fields_to_classify=args.fields,
                output_path=args.output
            )
            
            # Mostrar resultados
            console.print("[bold green]Resultados de la clasificación:[/bold green]")
            console.print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            # Mostrar instrucciones básicas
            console.print(Panel.fit(
                "Use --image para clasificar una imagen o --interactive para modo interactivo."
            ))
            parser.print_help()
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        logger.exception("Error en la ejecución")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
