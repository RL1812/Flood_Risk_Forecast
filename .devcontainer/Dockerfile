FROM mcr.microsoft.com/devcontainers/python:3.12

# 1. Install system tools, C libraries for geospatial packages (GDAL/PROJ/GEOS), and Node.js
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install global npm packages
RUN npm install -g antigravity

# 3. Copy requirements and install Python dependencies
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

# 4. install antigravity
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash