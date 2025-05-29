#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de OCR para el Sistema de Clasificación Documental Guiada.

Este módulo implementa la extracción de texto de imágenes y capturas de pantalla
utilizando DocTR como motor principal y EasyOCR como alternativa.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCREngine:
    """Motor de OCR para extraer texto de imágenes."""
    
    def __init__(self, engine: str = "doctr", lang: List[str] = None, device: str = None):
        """
        Inicializa el motor OCR.
        
        Args:
            engine: Motor OCR a utilizar ("doctr" o "easyocr")
            lang: Lista de idiomas para el OCR (por defecto ["es", "en"])
            device: Dispositivo para inferencia ("cpu" o "cuda")
        """
        self.engine = engine.lower()
        self.lang = lang or ["es", "en"]
        
        # Determinar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Iniciando OCR con motor {self.engine} en dispositivo {self.device}")
        
        # Inicializar el motor seleccionado
        if self.engine == "doctr":
            self._init_doctr()
        elif self.engine == "easyocr":
            self._init_easyocr()
        else:
            raise ValueError(f"Motor OCR no soportado: {engine}. Use 'doctr' o 'easyocr'")
    
    def _init_doctr(self):
        """Inicializa DocTR para OCR avanzado."""
        try:
            from doctr.models import ocr_predictor
            
            # Cargar modelo preentrenado
            self.model = ocr_predictor(pretrained=True, detect_language=True)
            
            # Mover modelo al dispositivo adecuado
            if self.device == "cuda":
                self.model.to(self.device)
                
            logger.info("DocTR inicializado correctamente")
            
        except ImportError:
            logger.error("No se pudo importar DocTR. Instale con: pip install python-doctr[torch]")
            raise
    
    def _init_easyocr(self):
        """Inicializa EasyOCR como alternativa."""
        try:
            import easyocr
            
            # Inicializar lector con los idiomas especificados
            self.model = easyocr.Reader(
                lang_list=self.lang,
                gpu=self.device == "cuda",
                quantize=False,  # Mejor calidad, a costa de velocidad
                verbose=False
            )
            
            logger.info("EasyOCR inicializado correctamente")
            
        except ImportError:
            logger.error("No se pudo importar EasyOCR. Instale con: pip install easyocr")
            raise
    
    def process_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Procesa una imagen para extraer texto.
        
        Args:
            image_path: Ruta a la imagen a procesar
            
        Returns:
            Dict: Diccionario con resultados del OCR, incluyendo:
                - text: Texto completo extraído
                - blocks: Bloques de texto con ubicaciones
                - confidence: Nivel de confianza
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
            
        logger.info(f"Procesando imagen: {image_path}")
        
        # Cargar imagen
        image = Image.open(image_path)
        
        if self.engine == "doctr":
            return self._process_with_doctr(image)
        else:
            return self._process_with_easyocr(image)
    
    def _process_with_doctr(self, image: Image.Image) -> Dict:
        """Procesa imagen con DocTR."""
        # Convertir a formato DocTR
        img = np.array(image.convert("RGB"))
        
        # Realizar OCR
        result = self.model([img])
        
        # Extraer texto completo
        full_text = ""
        blocks = []
        
        for page in result.pages:
            for block in page.blocks:
                block_text = ""
                for line in block.lines:
                    for word in line.words:
                        block_text += word.value + " "
                
                # Guardar bloque con coordenadas
                if block_text.strip():
                    blocks.append({
                        "text": block_text.strip(),
                        "bbox": block.geometry,  # Coordenadas normalizadas [xmin, ymin, xmax, ymax]
                        "confidence": block.confidence
                    })
                    full_text += block_text.strip() + "\n"
        
        return {
            "text": full_text.strip(),
            "blocks": blocks,
            "confidence": result.pages[0].confidence if result.pages else 0.0,
            "engine": "doctr"
        }
    
    def _process_with_easyocr(self, image: Image.Image) -> Dict:
        """Procesa imagen con EasyOCR."""
        # Convertir a formato EasyOCR
        img = np.array(image.convert("RGB"))
        
        # Realizar OCR con nivel de detalle
        result = self.model.readtext(img, detail=1, paragraph=True)
        
        # Extraer texto completo
        full_text = ""
        blocks = []
        total_confidence = 0.0
        
        for detection in result:
            coord, text, confidence = detection
            
            # Convertir a formato bbox [xmin, ymin, xmax, ymax]
            # EasyOCR retorna [[top-left], [top-right], [bottom-right], [bottom-left]]
            xmin = min(c[0] for c in coord) / img.shape[1]
            ymin = min(c[1] for c in coord) / img.shape[0]
            xmax = max(c[0] for c in coord) / img.shape[1]
            ymax = max(c[1] for c in coord) / img.shape[0]
            
            blocks.append({
                "text": text,
                "bbox": [xmin, ymin, xmax, ymax],
                "confidence": confidence
            })
            
            full_text += text + "\n"
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(result) if result else 0.0
        
        return {
            "text": full_text.strip(),
            "blocks": blocks,
            "confidence": avg_confidence,
            "engine": "easyocr"
        }
    
    def extract_structured_data(self, image_path: Union[str, Path]) -> Dict:
        """
        Extrae datos estructurados de la imagen, como tablas, coordenadas y etiquetas.
        
        Args:
            image_path: Ruta a la imagen a procesar
            
        Returns:
            Dict: Datos estructurados extraídos
        """
        # Esta función podría implementarse con lógica adicional para detectar
        # tablas, mapas, coordenadas, etc. basados en la distribución espacial
        # del texto y elementos visuales.
        
        # Por ahora, simplemente extraemos texto con metadata
        ocr_result = self.process_image(image_path)
        
        # Análisis simple de estructura basado en bloques
        table_candidates = []
        map_coordinates = []
        
        # Buscar patrones de tabla y coordenadas
        for block in ocr_result["blocks"]:
            text = block["text"]
            
            # Detectar posibles coordenadas (patrones como "lat: X, long: Y")
            if any(coord_pattern in text.lower() for coord_pattern in 
                   ["lat", "long", "coord", "gps", "ubicación", "°n", "°s", "°e", "°o"]):
                map_coordinates.append({
                    "text": text,
                    "bbox": block["bbox"]
                })
            
            # Detectar posibles filas de tabla (patrones de separación consistente)
            if "|" in text or "\t" in text or text.count(",") > 2:
                table_candidates.append({
                    "text": text,
                    "bbox": block["bbox"]
                })
        
        return {
            "text": ocr_result["text"],
            "tables": table_candidates,
            "coordinates": map_coordinates,
            "confidence": ocr_result["confidence"]
        }


# Función de ayuda para uso directo
def extract_text_from_image(
    image_path: Union[str, Path],
    engine: str = "doctr",
    lang: List[str] = None
) -> str:
    """
    Función auxiliar para extraer texto rápidamente de una imagen.
    
    Args:
        image_path: Ruta a la imagen
        engine: Motor OCR a utilizar ("doctr" o "easyocr")
        lang: Idiomas para reconocimiento
        
    Returns:
        str: Texto extraído de la imagen
    """
    ocr = OCREngine(engine=engine, lang=lang)
    result = ocr.process_image(image_path)
    return result["text"]


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        engine = sys.argv[2] if len(sys.argv) > 2 else "doctr"
        
        ocr = OCREngine(engine=engine)
        result = ocr.process_image(image_file)
        
        print(f"--- Texto extraído con {engine} ---")
        print(result["text"])
        print(f"Confianza: {result['confidence']:.2f}")
    else:
        print("Uso: python ocr.py <ruta_imagen> [doctr|easyocr]")
