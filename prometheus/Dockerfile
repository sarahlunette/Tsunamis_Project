# Utiliser une image Python de base
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'API dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de l'API
EXPOSE 8000

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "main:api", "--host", "localhost", "--port", "8000"]
