#!/bin/bash

# Set project root
PROJECT_ROOT="credit_scoring_classifier"
mkdir -p $PROJECT_ROOT && cd $PROJECT_ROOT

# Create directories
mkdir -p api
mkdir -p preprocessing
mkdir -p evaluation
mkdir -p model
mkdir -p observer
mkdir -p training
mkdir -p ui
mkdir -p utils
mkdir -p tests
mkdir -p model/artifacts

# Create Python init files
touch api/__init__.py
touch preprocessing/__init__.py
touch evaluation/__init__.py
touch model/__init__.py
touch observer/__init__.py
touch training/__init__.py
touch ui/__init__.py
touch utils/__init__.py
touch tests/__init__.py

# Create placeholders
touch main.py
touch requirements.txt
touch Dockerfile
touch docker-compose.yml
touch .gitignore
touch README.md

# Print done
echo "✅ Project scaffold created successfully under $PROJECT_ROOT/"
