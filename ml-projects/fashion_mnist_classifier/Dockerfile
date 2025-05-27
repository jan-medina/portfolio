# Dockerfile

FROM python:3.10-slim

# Crear carpeta de trabajo
WORKDIR /app

# Copiar requerimientos e instalar
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar código fuente
COPY . .

# Exponer puertos: API (8000), UI (8501)
EXPOSE 8000 8501

# Comando por defecto: ejecuta la API (se puede sobreescribir)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
