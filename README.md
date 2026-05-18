ArborID 🌲
AI-powered Tree Bark Classification with Deep Learning

ArborID is a Computer Vision project focused on recognizing tree species from bark images using Deep Learning and PyTorch.
The goal of the project is to build a scalable and production-style Machine Learning pipeline capable of handling real-world forest data, domain shifts, and challenging image conditions.

The project uses pretrained CNN architectures such as ResNet and applies modern training techniques including transfer learning, mixed precision training, weighted sampling, augmentation pipelines, and macro-F1 evaluation.

🚀 Features
🌳 Tree bark image classification
🧠 Transfer Learning with ResNet architectures
⚡ Mixed Precision Training (AMP)
📊 Macro-F1 and accuracy evaluation
🎯 Weighted loss for imbalanced datasets
🧪 Validation & inference pipeline
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
ResNet18 / ResNet50
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
│   ├── config.yaml
│   ├── train.yaml
│   ├── data.yaml
│   └── model.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── traindata/
│   ├── testdata/
│   └── cache/
│
├── src/
│   ├── data/
│   ├── trainer/
│   ├── model/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
│
├── outputs/
├── artifacts/
├── logs/
├── requirements.txt
└── README.md
🧠 How It Works

The model learns visual bark patterns such as:

texture
cracks
color distribution
shape irregularities
lighting behavior
micro-patterns in bark surfaces

Instead of manually defining features, the neural network automatically learns mathematical representations from images.

The workflow looks like this:

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

ArborID supports Automatic Mixed Precision (AMP) to accelerate training on NVIDIA GPUs.

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

The project focuses not only on accuracy but also on robust evaluation metrics:

Accuracy
Macro F1 Score
Top-K Accuracy
Confusion Analysis
Confidence Scores

Macro-F1 is especially important because bark datasets are often highly imbalanced.

🌍 Real-World Challenges

Tree bark classification is difficult because of:

changing weather
lighting conditions
camera differences
age of trees
moss and damage
domain shift between datasets

ArborID experiments with augmentation strategies and balancing methods to improve generalization.

🧪 Dataset

The project currently experiments with:

custom bark datasets
BarkNet dataset
manually filtered bark images
class-balanced subsets

Dataset structure:

dataset/
├── train/
│   ├── oak/
│   ├── pine/
│   └── birch/
│
├── validate/
└── test/
🚀 Running Training
Install dependencies
pip install -r requirements.txt
Start training
python src/train.py
Hydra override example
python src/train.py model=resnet50 optimizer=adamw
🧪 Evaluation

Run evaluation on folders:

python src/evaluation/eval_folder.py \
    --run_dir outputs/2026-01-01/12-00-00 \
    --data_dir data/testdata \
    --patches 20
📌 Current Goals
improve domain generalization
reduce dataset bias
test larger architectures
optimize inference pipeline
add explainability methods (Grad-CAM)
deploy API model service
experiment with Vision Transformers
🔮 Future Plans
🌐 Web application
📱 Mobile bark scanner
☁️ Cloud deployment
📡 Drone integration
🛰️ Forest monitoring systems
🤖 Active Learning pipeline
🎯 Real-time classification
📷 Example Use Cases
forestry automation
biodiversity monitoring
ecological research
educational tools
environmental AI systems
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
