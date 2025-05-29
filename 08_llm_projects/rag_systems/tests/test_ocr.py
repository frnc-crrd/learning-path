#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests para el módulo OCR del Sistema de Clasificación Documental Guiada.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# Importar módulo a testear
from app.ocr import OCREngine, extract_text_from_image


# Fixtures para pruebas
@pytest.fixture
def sample_image_path():
    """Crear una imagen de prueba temporal."""
    # Crear directorio para tests si no existe
    test_dir = Path("tests/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear una imagen simple con texto "TEST"
    image_path = test_dir / "test_image.png"
    
    # Crear imagen solo si no existe
    if not image_path.exists():
        # Crear imagen con PIL
        img = Image.new('RGB', (100, 30), color='white')
        
        # Sin necesidad de agregar texto real, solo simulamos
        img.save(image_path)
        
    return str(image_path)


@pytest.fixture
def mock_doctr_result():
    """Simular resultado de DocTR."""
    class MockPage:
        def __init__(self):
            self.blocks = []
            self.confidence = 0.9
            
    class MockBlock:
        def __init__(self, text):
            self.lines = []
            self.geometry = [0.1, 0.1, 0.9, 0.9]  # [xmin, ymin, xmax, ymax]
            self.confidence = 0.85
            
            # Crear línea con palabra
            line = MagicMock()
            word = MagicMock()
            word.value = text
            line.words = [word]
            self.lines = [line]
            
    class MockResult:
        def __init__(self):
            page = MockPage()
            block = MockBlock("Texto de prueba para OCR")
            page.blocks = [block]
            self.pages = [page]
            
    return MockResult()


@pytest.fixture
def mock_easyocr_result():
    """Simular resultado de EasyOCR."""
    # EasyOCR retorna: [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], text, confidence), ...]
    return [
        (
            [[10, 10], [90, 10], [90, 30], [10, 30]],
            "Texto de prueba para OCR",
            0.95
        )
    ]


# Tests
def test_ocr_engine_init_doctr():
    """Probar inicialización del motor DocTR."""
    with patch('app.ocr.ocr_predictor') as mock_ocr_predictor:
        engine = OCREngine(engine="doctr")
        assert engine.engine == "doctr"
        mock_ocr_predictor.assert_called_once()


def test_ocr_engine_init_easyocr():
    """Probar inicialización del motor EasyOCR."""
    with patch('app.ocr.easyocr') as mock_easyocr:
        # Configurar el mock del Reader
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader
        
        engine = OCREngine(engine="easyocr")
        assert engine.engine == "easyocr"
        mock_easyocr.Reader.assert_called_once()


def test_ocr_process_image_doctr(sample_image_path, mock_doctr_result):
    """Probar procesamiento de imagen con DocTR."""
    with patch('app.ocr.ocr_predictor') as mock_ocr_predictor:
        # Configurar el mock
        mock_model = MagicMock()
        mock_model.return_value = mock_doctr_result
        mock_ocr_predictor.return_value = mock_model
        
        # Crear instancia y procesar imagen
        engine = OCREngine(engine="doctr")
        
        # Reemplazar método real con mock
        engine.model = mock_model
        
        # Procesar imagen
        result = engine.process_image(sample_image_path)
        
        # Verificar resultados
        assert "text" in result
        assert "Texto de prueba para OCR" in result["text"]
        assert "blocks" in result
        assert result["engine"] == "doctr"
        assert "confidence" in result


def test_ocr_process_image_easyocr(sample_image_path, mock_easyocr_result):
    """Probar procesamiento de imagen con EasyOCR."""
    with patch('app.ocr.easyocr') as mock_easyocr:
        # Configurar el mock del Reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = mock_easyocr_result
        mock_easyocr.Reader.return_value = mock_reader
        
        # Crear instancia y procesar imagen
        engine = OCREngine(engine="easyocr")
        engine.model = mock_reader
        
        # Procesar imagen
        result = engine.process_image(sample_image_path)
        
        # Verificar resultados
        assert "text" in result
        assert "Texto de prueba para OCR" in result["text"]
        assert "blocks" in result
        assert result["engine"] == "easyocr"
        assert "confidence" in result


def test_extract_text_from_image(sample_image_path):
    """Probar función auxiliar de extracción de texto."""
    with patch('app.ocr.OCREngine') as MockOCREngine:
        # Configurar mock
        mock_engine_instance = MagicMock()
        mock_engine_instance.process_image.return_value = {
            "text": "Texto auxiliar extraído",
            "confidence": 0.9
        }
        MockOCREngine.return_value = mock_engine_instance
        
        # Ejecutar función
        result = extract_text_from_image(sample_image_path)
        
        # Verificar resultados
        assert result == "Texto auxiliar extraído"
        MockOCREngine.assert_called_once()
        mock_engine_instance.process_image.assert_called_once()


def test_invalid_image_path():
    """Probar comportamiento con ruta de imagen inválida."""
    engine = OCREngine()
    
    with pytest.raises(FileNotFoundError):
        engine.process_image("ruta/no/existente.png")
