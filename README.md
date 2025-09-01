# 🚦 Traffic Sign Recognition System

A professional Streamlit application for German traffic sign classification using advanced Convolutional Neural Networks (CNN).

**Created by Hamza Younas | Elevvo Tech**

![Traffic Sign Recognition](https://img.shields.io/badge/Deep%20Learning-Traffic%20Sign%20Recognition-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=opencv&logoColor=white)

## 🚀 Live Demo

🌐 **[Access the Application](https://traffic-sign-recognition-system-by-hamza.streamlit.app/)**

## Features

### 🔍 Image Classification
- Real-time traffic sign classification with 99.53% accuracy
- Confidence scoring for each prediction
- Support for 43 different German traffic sign classes
- Image upload with instant AI analysis

### 📊 Model Evaluation
- Comprehensive performance analysis with confusion matrix
- Per-class accuracy metrics and detailed classification reports
- Interactive visualizations with matplotlib and seaborn
- Test dataset evaluation (50 samples per class)

### 📚 Project Information
- Complete technical specifications and model architecture
- Performance metrics and training details
- Technology stack and development information

### 🎨 Modern Interface
- Clean and professional design without excessive emojis
- Responsive layout with beautiful animations
- Glass-morphism effects and gradient backgrounds
- Navigation sidebar for easy page switching

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   - Download the GTSRB dataset
   - Extract to folder: `gtsrb german trafficsign/`
   - Ensure the folder structure matches: `Train/0/`, `Train/1/`, etc.

3. **Train the Model**
   - Open and run all cells in `traffic_sign_recognition.ipynb`
   - This will create `traffic_sign_model.h5` and `class_names.pickle`

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Model Architecture

- **Deep CNN**: Multi-layer Convolutional Neural Network
- **Batch Normalization**: For training stability
- **Dropout Layers**: For regularization and preventing overfitting
- **Input Size**: 32x32 RGB images
- **Output**: 43 traffic sign classes
- **Parameters**: 875,083 trainable parameters

## Performance Metrics

- **Test Accuracy**: 99.53%
- **Training Framework**: TensorFlow/Keras
- **Optimizer**: Adam with learning rate scheduling
- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Training Samples**: 6,450 images (150 per class)

## Technologies

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision operations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning utilities
- **NumPy/Pandas**: Data manipulation
- **Pillow**: Image processing

## Application Structure

```
├── app.py                              # Main Streamlit application
├── traffic_sign_recognition.ipynb      # Complete ML pipeline
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── traffic_sign_model.h5              # Trained CNN model
├── class_names.pickle                 # Traffic sign class mappings
└── gtsrb german trafficsign/          # GTSRB dataset (user provided)
    ├── Train/                         # Training images by class
    │   ├── 0/                        # Speed limit 20km/h
    │   ├── 1/                        # Speed limit 30km/h
    │   └── ...                       # Additional classes
    └── Test/                         # Test images
```

## Traffic Sign Classes

The system can classify 43 different German traffic sign types including:
- Speed limit signs (20-120 km/h)
- Warning signs (curves, road work, pedestrians)
- Regulatory signs (no passing, stop, yield)
- Mandatory signs (turn directions, roundabout)

## Usage

1. **Navigate** through sections using the sidebar:
   - **Image Classification**: Upload and classify traffic signs
   - **Model Evaluation**: View performance metrics and confusion matrix
   - **About**: Technical details and project information

2. **Upload Images**: Drag and drop traffic sign images for instant classification

3. **View Results**: Get predicted class, sign type, and confidence score

4. **Analyze Performance**: Run model evaluation to see detailed metrics

## Model Training Details

### Data Preprocessing
- Image resizing to 32x32 pixels
- Pixel normalization (0-1 range)
- Train/validation/test split (64%/16%/20%)

### Training Configuration
- **Epochs**: 30 with early stopping
- **Batch Size**: 32
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Callbacks**: Early stopping, learning rate reduction

### Architecture Highlights
- 3 Convolutional blocks with BatchNormalization
- MaxPooling and Dropout for regularization
- Dense layers with 512 and 256 neurons
- Softmax output for 43-class classification

## 📊 Screenshots

<img width="1478" height="1317" alt="download" src="https://github.com/user-attachments/assets/7fc17134-d3ad-44d4-9fbf-f6635063749a" />
<img width="954" height="437" alt="msedge_IJdTB2Kw34" src="https://github.com/user-attachments/assets/72450d83-106a-4a92-ac10-82651da65374" />


## 🙏 Acknowledgments

- **German Traffic Sign Recognition Benchmark (GTSRB)** for the comprehensive dataset
- **Streamlit team** for the amazing web framework
- **TensorFlow team** for the powerful deep learning tools

## 📧 Contact

**Hamza Younas**
- GitHub: [@Hamzaviour](https://github.com/Hamzaviour)
- LinkedIn: [Hamza Younas](https://linkedin.com/in/hamza-younas)
- Email: hamzavelous@gmail.com

---

🚦 **Built with ❤️ using Deep Learning and Computer Vision | Elevvo Tech**
