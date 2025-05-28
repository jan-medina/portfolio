#!/bin/bash

# name of base project
PROJECT_NAME="fashion_mnist_classifier"

# Creates main folders
mkdir -p $PROJECT_NAME/{data,model,training,evaluation,prediction,utils,notebooks}

# Creates empty python files
touch $PROJECT_NAME/data/data_loader.py
touch $PROJECT_NAME/model/{builder.py,cnn_model.py}
touch $PROJECT_NAME/training/{engine.py,callbacks.py}
touch $PROJECT_NAME/evaluation/{metrics.py,visualizer.py}
touch $PROJECT_NAME/prediction/predictor.py
touch $PROJECT_NAME/utils/{config.py,logger.py}
touch $PROJECT_NAME/main.py
touch $PROJECT_NAME/README.md

# Requirements files
cat <<EOL > $PROJECT_NAME/requirements.txt
torch
torchvision
matplotlib
seaborn
scikit-learn
numpy
pandas
EOL

echo "Project base dir structure $PROJECT_NAME was sucessfully created 🚀"
