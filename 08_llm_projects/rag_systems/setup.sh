#!/bin/bash
# Script de instalación para el Sistema de Clasificación Documental Guiada (RAG)

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Instalando Sistema de Clasificación Documental Guiada...${NC}"

# Verificar Python
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )([0-9]+\.[0-9]+\.[0-9]+)')
required_version="3.9.0"

if [ -z "$python_version" ]; then
    echo -e "${RED}Error: Python 3 no encontrado. Por favor, instale Python 3.9+${NC}"
    exit 1
fi

# Comparar versiones
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)
required_major=$(echo $required_version | cut -d. -f1)
required_minor=$(echo $required_version | cut -d. -f2)

if [ "$python_major" -lt "$required_major" ] || ([ "$python_major" -eq "$required_major" ] && [ "$python_minor" -lt "$required_minor" ]); then
    echo -e "${RED}Error: Se requiere Python 3.9+. Versión actual: $python_version${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $python_version encontrado${NC}"

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error al crear entorno virtual${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}El entorno virtual ya existe${NC}"
fi

# Activar entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Error al activar entorno virtual${NC}"
    exit 1
fi

# Instalar dependencias
echo "Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Error al instalar dependencias${NC}"
    exit 1
fi

# Crear directorios necesarios si no existen
echo "Verificando estructura de directorios..."
mkdir -p data/guidelines data/samples

# Verificar Ollama (opcional)
echo -e "${YELLOW}Verificando Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama no encontrado. Para instalar Ollama, visite: https://ollama.ai/${NC}"
    echo -e "${YELLOW}Una vez instalado, ejecute: ollama pull mistral:7b${NC}"
else
    echo -e "${GREEN}✓ Ollama encontrado${NC}"
    # Verificar modelo Mistral
    echo "Verificando modelo Mistral..."
    if ! ollama list | grep -q "mistral:7b"; then
        echo "Descargando modelo Mistral 7B (esto puede tomar tiempo)..."
        ollama pull mistral:7b
    else
        echo -e "${GREEN}✓ Modelo Mistral:7b encontrado${NC}"
    fi
fi

echo -e "${GREEN}¡Instalación completada con éxito!${NC}"
echo -e "Para activar el entorno virtual: ${YELLOW}source venv/bin/activate${NC}"
echo -e "Para ejecutar el sistema: ${YELLOW}python app/main.py --help${NC}"
