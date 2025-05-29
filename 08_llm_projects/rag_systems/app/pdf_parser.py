#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para procesar y extraer texto de documentos PDF.

Este módulo implementa funcionalidades para extraer texto estructurado de documentos PDF,
dividirlo en chunks semánticos y mantener referencias a páginas y secciones.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Generator, Any, Union
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """Fragmento de texto extraído de un PDF con metadatos."""
    
    text: str
    page_number: int
    chunk_id: str
    heading: str = ""
    section: str = ""
    subsection: str = ""
    bbox: Tuple[float, float, float, float] = None  # [x0, y0, x1, y1]
    

@dataclass
class PDFDocument:
    """Representación de un documento PDF procesado."""
    
    path: str
    title: str = ""
    author: str = ""
    num_pages: int = 0
    toc: List[Dict] = field(default_factory=list)
    chunks: List[PDFChunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def get_chunks_for_page(self, page_num: int) -> List[PDFChunk]:
        """Obtener todos los chunks de una página específica."""
        return [chunk for chunk in self.chunks if chunk.page_number == page_num]
    
    def get_section_chunks(self, section: str) -> List[PDFChunk]:
        """Obtener todos los chunks de una sección específica."""
        return [chunk for chunk in self.chunks if chunk.section == section]
    
    def get_document_text(self) -> str:
        """Obtener texto completo del documento."""
        return "\n\n".join([chunk.text for chunk in self.chunks])


class PDFParser:
    """
    Parser para extraer texto y metadatos de archivos PDF.
    
    Este parser utiliza PyMuPDF (fitz) para extraer contenido
    estructurado de PDFs, preservando la estructura y metadatos.
    """
    
    def __init__(self, 
                 min_chunk_size: int = 200, 
                 max_chunk_size: int = 1000,
                 overlap: int = 50):
        """
        Inicializa el parser PDF.
        
        Args:
            min_chunk_size: Tamaño mínimo de un chunk en caracteres
            max_chunk_size: Tamaño máximo de un chunk en caracteres
            overlap: Superposición entre chunks consecutivos
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.headings = {}  # Mapeo de página -> encabezados
    
    def parse_pdf(self, pdf_path: Union[str, Path]) -> PDFDocument:
        """
        Procesa un archivo PDF y extrae su contenido estructurado.
        
        Args:
            pdf_path: Ruta al archivo PDF a procesar
            
        Returns:
            PDFDocument: Documento procesado con chunks y metadatos
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Archivo PDF no encontrado: {pdf_path}")
        
        logger.info(f"Procesando PDF: {pdf_path}")
        
        # Crear documento PDF
        doc = fitz.open(pdf_path)
        
        # Extraer metadatos básicos
        metadata = doc.metadata
        toc = doc.get_toc()
        
        pdf_document = PDFDocument(
            path=str(pdf_path),
            title=metadata.get("title", pdf_path.stem),
            author=metadata.get("author", ""),
            num_pages=doc.page_count,
            toc=self._process_toc(toc),
            metadata=metadata
        )
        
        # Extraer encabezados para mapeo de secciones
        self._extract_headings(doc)
        
        # Procesar páginas y generar chunks
        logger.info(f"Extrayendo contenido de {doc.page_count} páginas")
        for page_num in tqdm(range(doc.page_count), desc="Procesando páginas"):
            page = doc[page_num]
            page_chunks = self._process_page(page, page_num)
            pdf_document.chunks.extend(page_chunks)
        
        logger.info(f"PDF procesado. Generados {len(pdf_document.chunks)} chunks")
        return pdf_document
    
    def _extract_headings(self, doc: fitz.Document) -> None:
        """
        Extrae los encabezados del documento para mapearlos a páginas.
        
        Args:
            doc: Documento PDF abierto
        """
        self.headings = {}
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            
            # Extraer bloques de texto
            blocks = page.get_text("dict")["blocks"]
            
            page_headings = []
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Detectar posibles encabezados por tamaño de fuente y estilo
                        if span["size"] > 12 or span["flags"] & 16:  # 16 = bold
                            text = span["text"].strip()
                            if text and len(text) < 100:  # Evitar texto largo como encabezado
                                page_headings.append({
                                    "text": text,
                                    "bbox": span["bbox"],
                                    "font_size": span["size"],
                                    "is_bold": bool(span["flags"] & 16)
                                })
            
            if page_headings:
                self.headings[page_idx] = page_headings
    
    def _get_section_for_chunk(self, page_num: int, y_pos: float) -> Tuple[str, str]:
        """
        Determina la sección y subsección para un chunk basado en posición Y.
        
        Args:
            page_num: Número de página
            y_pos: Posición vertical del chunk
            
        Returns:
            Tuple[str, str]: (sección, subsección)
        """
        section, subsection = "", ""
        
        if page_num in self.headings:
            # Encontrar el encabezado más cercano por encima del chunk
            for heading in sorted(self.headings[page_num], key=lambda h: h["bbox"][1]):
                if heading["bbox"][1] < y_pos:
                    if heading["font_size"] >= 14 or heading["is_bold"]:
                        section = heading["text"]
                    else:
                        subsection = heading["text"]
        
        return section, subsection
    
    def _process_page(self, page: fitz.Page, page_num: int) -> List[PDFChunk]:
        """
        Procesa una página y la divide en chunks.
        
        Args:
            page: Página del documento PDF
            page_num: Número de página (0-indexed)
            
        Returns:
            List[PDFChunk]: Lista de chunks extraídos de la página
        """
        # Extraer texto de la página
        text = page.get_text("text")
        if not text.strip():
            return []
        
        # Dividir en párrafos naturales
        paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # Agrupar párrafos en chunks
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Si el chunk actual ya es suficientemente grande, guardarlo
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    # Crear chunk
                    chunk_id = f"page_{page_num + 1}_{len(chunks) + 1}"
                    section, subsection = self._get_section_for_chunk(page_num, 0)  # Mejorable con posición real
                    
                    chunks.append(PDFChunk(
                        text=current_chunk.strip(),
                        page_number=page_num + 1,  # 1-indexed para usuario final
                        chunk_id=chunk_id,
                        heading=section,
                        section=section,
                        subsection=subsection
                    ))
                
                # Iniciar nuevo chunk con superposición si es necesario
                current_chunk = para + "\n\n"
        
        # No olvidar el último chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_id = f"page_{page_num + 1}_{len(chunks) + 1}"
            section, subsection = self._get_section_for_chunk(page_num, 0)
            
            chunks.append(PDFChunk(
                text=current_chunk.strip(),
                page_number=page_num + 1,
                chunk_id=chunk_id,
                heading=section,
                section=section,
                subsection=subsection
            ))
        
        return chunks
    
    def _process_toc(self, toc: List) -> List[Dict]:
        """
        Procesa la tabla de contenido para formato uniforme.
        
        Args:
            toc: Tabla de contenido extraída de PyMuPDF
            
        Returns:
            List[Dict]: Tabla de contenido procesada
        """
        processed_toc = []
        
        for item in toc:
            if len(item) >= 3:
                level, title, page = item[:3]
                processed_toc.append({
                    "level": level,
                    "title": title,
                    "page": page
                })
        
        return processed_toc


# Función auxiliar para uso directo
def extract_pdf_content(
    pdf_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> PDFDocument:
    """
    Función auxiliar para extraer contenido de un PDF.
    
    Args:
        pdf_path: Ruta al archivo PDF
        output_dir: Directorio para guardar resultados (opcional)
        
    Returns:
        PDFDocument: Documento procesado
    """
    parser = PDFParser()
    document = parser.parse_pdf(pdf_path)
    
    # Guardar chunks si se especifica directorio
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{Path(pdf_path).stem}_chunks.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(document.chunks):
                f.write(f"--- CHUNK {i+1} (Página {chunk.page_number}) ---\n")
                f.write(f"Sección: {chunk.section}\n")
                f.write(f"Subsección: {chunk.subsection}\n")
                f.write(f"Texto:\n{chunk.text}\n\n")
                f.write("-" * 80 + "\n\n")
    
    return document


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        document = extract_pdf_content(pdf_file, output_dir)
        
        print(f"--- Documento: {document.title} ---")
        print(f"Autor: {document.author}")
        print(f"Páginas: {document.num_pages}")
        print(f"Chunks extraídos: {len(document.chunks)}")
        
        # Mostrar primeros chunks
        for i, chunk in enumerate(document.chunks[:3]):
            print(f"\n--- CHUNK {i+1} (Página {chunk.page_number}) ---")
            print(f"Sección: {chunk.section}")
            print(chunk.text[:150] + "...")
    else:
        print("Uso: python pdf_parser.py <ruta_pdf> [directorio_salida]")
