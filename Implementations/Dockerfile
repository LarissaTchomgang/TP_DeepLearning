
# 1) Image de base : Python 3.10

FROM python:3.10-slim


# 2) Installer les dépendances système
#    (TensorFlow en a absolument besoin)

RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


# 3) Créer un dossier pour ton app

WORKDIR /app


# 4) Copier seulement requirements.txt d'abord
#    (pour profiter du cache Docker)

COPY requirements.txt .


# 5) Installer les dépendances Python

RUN pip install --no-cache-dir -r requirements.txt


# 6) Copier tout le reste du projet

COPY . .

# 7) Lancer l'application Flask

CMD ["python", "app.py"]
