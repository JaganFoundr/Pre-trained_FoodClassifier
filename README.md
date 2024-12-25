# 🍕 Sushi-Steak Classifier Using PyTorch  

Welcome to the **Pizza-Sushi-Steak Classifier**! This project demonstrates how to use **PyTorch** and its pretrained models to classify images of **pizza**, **sushi**, and **steak** with a small dataset of just 300 images.  

---

## 🚀 Project Overview  

The goal of this project was to identify the best model architecture for classifying images into these three food categories using a highly constrained dataset:  
- **Dataset Size**:  
  - **225 images** for training  
  - **75 images** for testing  

Through experimentation, we evaluated multiple models, including **ResNet**, **EfficientNet**, and **Vision Transformers (ViT)**.  

---

## 🔑 Key Features  

- **Pretrained PyTorch Models**: Utilized ResNet, EfficientNet, and Vision Transformers to leverage transfer learning.  
- **Feature Extraction**: Explored the effectiveness of using pretrained layers without retraining the full model.  
- **Compact Dataset**: Tackled the challenges of training on a small dataset.  
- **Vision Transformers (ViT)**: Achieved the best results using **ViT-B_32** and **ViT-B_16** as feature extractors.  

---

## 🧪 Models Explored  

### 1️⃣ **ResNet Series**  
- **Tested Models**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152  
- **Outcome**: Limited results due to dataset size and hardware constraints.  

### 2️⃣ **EfficientNet Series**  
- **Tested Models**: EfficientNet-B0, B1, B2  
- **Outcome**: EfficientNet-B2 (Feature Extractor) delivered reasonable results.  

### 3️⃣ **Vision Transformers (ViT)**  
- **Tested Models**: ViT-B_32, ViT-B_16  
- **Outcome**: As feature extractors, they delivered the **best performance**, showing the power of transformer-based architectures in computer vision.  

---

## 📂 Project Structure  

```plaintext  
root/  
├── dataset/         # Contains training and testing images  
├── models/          # Saved PyTorch models  
├── README.md        # Project documentation (this file)  
```

---

## 🧑‍💻 Results  

- **Best Model**: Vision Transformer (ViT-B_16 as a feature extractor)  
- **Accuracy**: Achieved 85% accuracy on the test set.  
- **Inference Time**: Lightweight and fast for real-time applications.  

---

## 🌟 Highlights  

- **Transformers in Vision**: Demonstrates how Vision Transformers can outperform traditional CNNs on small datasets.  
- **Open for Improvement**: This project is designed to be a starting point—fine-tune the models, explore other architectures, or expand the dataset.  

---

## 🤝 Contributions  

Contributions are welcome! If you’d like to improve this project, please fork the repository and create a pull request.  

---

## 📜 License  

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.  

---

## 💬 Let’s Connect  

If you found this project useful or have any questions, feel free to reach out or share your feedback:  
- **LinkedIn**: [Jagannath](www.linkedin.com/in/jaganfoundr)  
- **GitHub**: [JaganFoundr](https://github.com/JaganFoundr)  
