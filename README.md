ArborID 🌲
AI-powered Tree Bark Classification with Deep Learning

ArborID is a Computer Vision project focused on tree species recognition from bark images using Deep Learning and PyTorch.

The project was designed to simulate a real-world Machine Learning pipeline capable of handling:

domain shift
imbalanced datasets
different camera devices
challenging outdoor lighting conditions
real forest environments

ArborID uses pretrained Convolutional Neural Networks such as ResNet and combines them with modern Deep Learning techniques including:

Transfer Learning
Mixed Precision Training (AMP)
Weighted Loss Functions
Data Augmentation
Macro-F1 evaluation
Early Stopping
Modular Training Pipelines
🚀 Features
🌳 Tree bark image classification
🧠 Transfer Learning with ResNet architectures
⚡ Mixed Precision Training (AMP)
📊 Macro-F1 and accuracy evaluation
🎯 Weighted loss for imbalanced datasets
🧪 Validation and inference pipeline
🔥 Early stopping support
🧰 Hydra configuration system
📁 Modular project structure
📈 Training history logging
🧹 Dataset preprocessing utilities
🧪 Multi-patch inference experiments
🐍 PyTorch-based training ecosystem
🛠️ Technologies
Core
Python
PyTorch
Torchvision
Hydra
OmegaConf
Machine Learning
ResNet18
ResNet50
Transfer Learning
Cross Entropy Loss
AdamW
SGD
Learning Rate Scheduling
Macro F1 Score
Weighted Sampling
Utilities
NumPy
Pillow
tqdm
scikit-learn
matplotlib
📂 Project Structure
ArborID/
│
├── configs/
│   ├── data_barknet.yaml
│   ├── data.yaml
│   └── train.yaml
│
├── src/
│   ├── data/
│   │   ├── testdata/
│   │   ├── traindata/
│   │   ├── data_barknet.py
│   │   ├── loader.py
│   │   ├── prep_data.py
│   │   ├── schemas.py
│   │   └── transforms.py
│   │
│   ├── evaluation/
│   │   ├── eval_folder.py
│   │   ├── evaluation.py
│   │   └── visualization.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   ├── trainer/
│   │   ├── early_stopping.py
│   │   ├── infer.py
│   │   ├── train.py
│   │   └── utils.py
│   │
│   ├── train.py
│   ├── utils.py
│   └── __init__.py
│
├── eval/
├── outputs/
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
🧠 How It Works

The model learns visual bark patterns such as:

texture
cracks
bark geometry
color distribution
lighting behavior
micro-patterns on bark surfaces

Instead of manually defining features, the neural network automatically learns mathematical representations from image data.

Pipeline overview:

Image → CNN Backbone → Feature Extraction → Classification Head → Tree Species
📊 Training Pipeline

The training pipeline includes:

image preprocessing
augmentations
dataset balancing
forward propagation
backpropagation
optimizer updates
validation
checkpoint saving

Example augmentations:

RandomResizedCrop
HorizontalFlip
ColorJitter
RandomRotation
Normalize
⚡ Mixed Precision Training

ArborID supports Automatic Mixed Precision (AMP) for faster GPU training.

Benefits:

faster training
lower VRAM usage
larger batch sizes
Tensor Core acceleration

Example:

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
📈 Metrics

The project focuses on robust evaluation metrics instead of accuracy alone.

Implemented metrics:

Accuracy
Macro F1 Score
Top-K Accuracy
Confidence Scores
Confusion Analysis

Macro-F1 is especially important because bark datasets are highly imbalanced.

🌍 Real-World Challenges

Tree bark classification is difficult because of:

weather conditions
camera differences
lighting changes
tree age
moss and damage
dataset domain shift

ArborID experiments with augmentation strategies and balancing methods to improve generalization on unseen data.

🧪 Dataset

The project currently experiments with:

BarkNet dataset
custom bark datasets
manually filtered bark images
class-balanced subsets

Current dataset structure:

src/data/
├── traindata/
├── testdata/
🚀 Running Training
Install dependencies
pip install -r requirements.txt
Start training
python src/train.py
Hydra override example
python src/train.py optimizer=adamw
🧪 Evaluation

Run evaluation on folders:

python src/evaluation/eval_folder.py \
    --run_dir outputs/2026-01-01/12-00-00 \
    --data_dir src/data/testdata \
    --patches 20
📌 Current Goals
improve domain generalization
reduce dataset bias
optimize inference pipeline
test larger architectures
add explainability methods (Grad-CAM)
experiment with Vision Transformers
deploy model APIs
🔮 Future Plans
🌐 Web application
📱 Mobile bark scanner
☁️ Cloud deployment
📡 Drone integration
🛰️ Forest monitoring systems
🤖 Active Learning pipelines
🎯 Real-time classification
📷 Example Use Cases
forestry automation
biodiversity monitoring
ecological research
educational AI tools
environmental monitoring
autonomous forest scanning
🤝 Contributing

Contributions, experiments and suggestions are welcome.

git fork
git clone
git commit
git push
📜 License

MIT License

👨‍💻 Author
Emil Nowak

Machine Learning Engineer focused on:

Computer Vision
Deep Learning
AI Systems
PyTorch
Forest AI Research

GitHub:
ReXeR00 GitHub