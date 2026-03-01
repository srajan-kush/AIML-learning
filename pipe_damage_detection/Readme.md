# 🚀 Pipe Damage Detection using CNN (Transfer Learning - ResNet18)

An end-to-end Deep Learning based web application that classifies industrial pipe images as **Damaged** or **Undamaged** using Transfer Learning (ResNet18).

This project demonstrates:

- ✅ CNN-based image classification  
- ✅ Transfer learning using ResNet18  
- ✅ Automatic dataset splitting  
- ✅ Model training & evaluation  
- ✅ Flask-based web deployment  
- ✅ Confidence scoring  
- ✅ Error handling  
- ✅ Clean UI  

---

# 📌 Problem Statement

In industrial environments (Oil & Gas, Manufacturing, Infrastructure), manual inspection of pipes for cracks and defects is:

- Time-consuming  
- Expensive  
- Error-prone  

This project builds an AI-powered system that automatically detects whether a pipe is **damaged or undamaged** using computer vision.

---

# 🧠 Model Architecture

## 🔹 Base Model: ResNet18 (Transfer Learning)

We use a pretrained ResNet18 model trained on ImageNet and fine-tune it for binary classification.

### Why ResNet18?

- Prevents vanishing gradient problem using residual connections  
- Works well on small datasets  
- Faster training compared to deeper networks  
- High generalization ability  


---

# 🏗 Project Structure

```
pipe_damage_detection/
│
├── data/
│   ├── damaged/
│   └── undamaged/
│
├── models/
│   └── best_model.pth
│
├── static/
├── templates/
│   └── index.html
│
├── train.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📊 Dataset

Source: Kaggle Pipe Images Dataset  

Classes:
- Damaged
- Undamaged  

Dataset is automatically split:
- 80% Training  
- 20% Validation  

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone <check the url>
cd pipe_damage_detection
```

---

## 2️⃣ Create Virtual Environment (Recommended)

```bash
conda create -n pipe python=3.10
conda activate pipe
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training the Model

Run:

```bash
python train.py
```

Training Process:

- Loads dataset
- Applies augmentation
- Fine-tunes ResNet18
- Saves best model in `models/`
- Prints validation accuracy

Expected Accuracy:
~85%–95% depending on dataset split

---

# 🌐 Running the Web Application

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

Upload an image → Get:

- Prediction result  
- Confidence score  
- Uploaded image preview  
- Error handling  

---

# 🧪 Features Implemented

✔ Transfer Learning  
✔ Automatic Dataset Split  
✔ Confidence Percentage  
✔ Modern UI  
✔ File Type Validation  
✔ Model Save & Load  
✔ Folder Auto-Creation  
✔ CPU Compatible  

---

# 🔍 Error Handling

The application handles:

- No file uploaded  
- Empty filename  
- Unsupported file formats  
- Missing directories  
- Model loading issues  
- Prediction runtime errors  

---

# 📈 Performance

Typical Validation Accuracy:
- ~90% after 5–10 epochs  

Inference Speed:
- Real-time on CPU  

---

# 💼 Industrial Applications

- Oil & Gas pipeline inspection  
- Infrastructure maintenance  
- Smart manufacturing  
- Automated defect detection  
- Quality control automation  

---

# 🚀 Future Improvements

- Grad-CAM visualization  
- FastAPI deployment  
- Docker containerization  
- AWS/GCP cloud deployment  
- Real-time webcam detection  
- Model quantization for edge devices  
- CI/CD pipeline integration  

---

# 🧑‍💻 Author

**Srajan Kushwaha**  
AI/ML Enthusiast | Full Stack Developer  
IIITDM Kurnool  

---

# 📜 License

This project is created for educational and research purposes.

---

# ⭐ If You Found This Useful

Give this repository a ⭐ on GitHub.