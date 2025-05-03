# 🎵 Music Genre Classification & Recommendation App

This project focuses on exploring audio data, classifying music genres using machine learning models (CNN & XGBoost), and building a basic music recommendation system. The entire workflow is demonstrated in a Jupyter Notebook and deployable using **Streamlit**.

---

## 📌 Project Overview

- 📂 **Dataset**: [GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- 🎧 10 music genres with 100 audio files each (30 seconds per file)
- 📊 Data represented via **Mel Spectrograms** for deep learning
- 🤖 Models used:
  - **CNN (Convolutional Neural Network)** for spectrogram images
  - **XGBoost** for handcrafted feature-based classification
- 🎯 Includes a basic **Music Recommendation System** based on genre similarity

---

## 🚀 Features

- 🎼 Audio Data Exploration using `librosa`
- 📉 Visualization of audio features: waveform, MFCCs, Mel Spectrogram
- 🔍 Genre classification using:
  - **Convolutional Neural Network (CNN)**
  - **XGBoost Classifier**
- 🧠 Performance evaluation with accuracy and confusion matrix
- 🔄 Basic rule-based recommendation engine
- 🌐 Deployable via **Streamlit** for interactive user experience

---

## 🛠️ Technologies Used

- Python 3.8+
- `librosa`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `tensorflow/keras`
- `pandas`, `numpy`
- `Streamlit` (for app deployment)

---

## 📥 Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/music-genre-classification.git
cd music-genre-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

To explore the data and model pipeline:

```bash
jupyter notebook music_genre_classification_streamlit.ipynb
```

### 4. Launch the Streamlit App

If the notebook has been converted into a Python script (e.g. `app.py`):

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
music-genre-classification/
│
├── music_genre_classification_streamlit.ipynb   # Main notebook
├── app.py                                       # (Optional) Streamlit version
├── data/                                        # Audio files and metadata
├── models/                                      # Trained models (if saved)
├── README.md
└── requirements.txt
```

---

## 👨‍💻 Authors

- Trần Trọng Kiên - 22110093  
- Châu Gia Kiệt - 22110095  
- Trương Hồng Kiệt - 22110096  
- Trương Minh Quân - 22110172

---

## 📄 License

This project is created for educational purposes only.
