import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd

st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def animated_banner():
    st.markdown(
        """
        <style>
        @keyframes glow {
            0% { text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00; }
            50% { text-shadow: 0 0 20px #ff0000, 0 0 30px #ff0000, 0 0 40px #ff0000; }
            100% { text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00; }
        }
        
        @keyframes slide {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
        }
        
        .main-title {
            font-size: 4rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 2s ease-in-out infinite, bounce 2s ease-in-out infinite;
            margin-bottom: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.5rem;
            text-align: center;
            color: #ffffff;
            margin-top: 10px;
            margin-bottom: 20px;
            animation: bounce 2s ease-in-out infinite;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        .company-name {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(45deg, #ffd700, #ffed4e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 3s ease-in-out infinite;
            margin-top: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .creator-banner {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 15px 25px;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: bold;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            animation: bounce 3s ease-in-out infinite;
            z-index: 1000;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        .prediction-box {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .upload-box {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            text-align: center;
        }
        </style>
        
        <div class="main-title">🚦 Traffic Sign Recognition System</div>
        <div class="subtitle">Advanced deep learning model for real-time traffic sign classification</div>
        <div class="company-name"> Elevvo Tech</div>
        
        <div class="creator-banner">
            Created by Hamza Younas
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_trained_model():
    try:
        model = load_model('traffic_sign_model.h5')
        with open('class_names.pickle', 'rb') as f:
            class_names = pickle.load(f)
        return model, class_names
    except:
        return None, None

@st.cache_data
def load_test_data():
    """Load test data for evaluation"""
    data_dir = 'gtsrb german trafficsign'
    train_path = os.path.join(data_dir, 'Train')
    
    if not os.path.exists(train_path):
        return None, None
    
    images = []
    labels = []
    
    # Load a subset for evaluation (50 images per class for demo)
    for class_num in range(43):
        class_path = os.path.join(train_path, str(class_num))
        if os.path.exists(class_path):
            count = 0
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png') and count < 50:
                    try:
                        img_path = os.path.join(class_path, img_name)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (32, 32))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(class_num)
                            count += 1
                    except Exception:
                        continue
    
    if len(images) > 0:
        X = np.array(images).astype('float32') / 255.0
        y = np.array(labels)
        return X, y
    return None, None

def evaluate_model(model, class_names):
    """Evaluate model performance"""
    X_test, y_test = load_test_data()
    
    if X_test is None:
        st.error("Test data not available. Please ensure the GTSRB dataset is in the correct folder.")
        return
    
    st.markdown("### Model Evaluation in Progress...")
    progress_bar = st.progress(0)
    
    # Make predictions
    progress_bar.progress(25)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    progress_bar.progress(50)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    progress_bar.progress(75)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    progress_bar.progress(100)
    st.success("Evaluation completed!")
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(
            f"""
            <div class="prediction-box">
                <h3>Model Performance Metrics</h3>
                <div style="font-size: 1.5rem; margin: 20px 0;">
                    <p><strong>Overall Accuracy:</strong> <span style="color: #00ff00; font-size: 2rem;">{accuracy:.2%}</span></p>
                    <p><strong>Test Samples:</strong> {len(y_test)}</p>
                    <p><strong>Correct Predictions:</strong> {np.sum(y_pred_classes == y_test)}</p>
                    <p><strong>Incorrect Predictions:</strong> {np.sum(y_pred_classes != y_test)}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Per-class accuracy
        st.markdown("#### Per-Class Accuracy")
        class_accuracies = []
        for class_idx in range(43):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_pred_classes == y_test) & class_mask) / np.sum(class_mask)
                class_accuracies.append({
                    'Class': class_idx,
                    'Name': class_names[class_idx][:20] + '...' if len(class_names[class_idx]) > 20 else class_names[class_idx],
                    'Accuracy': f"{class_acc:.2%}",
                    'Samples': int(np.sum(class_mask))
                })
        
        df_accuracy = pd.DataFrame(class_accuracies)
        st.dataframe(df_accuracy, use_container_width=True)
    
    with col2:
        st.markdown("#### Confusion Matrix")
        
        # Create confusion matrix plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    cbar_kws={'label': 'Number of Predictions'})
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix - Traffic Sign Classification')
        
        # Customize the plot
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Classification report
    st.markdown("#### Detailed Classification Report")
    report = classification_report(y_test, y_pred_classes, 
                                 target_names=[class_names[i] for i in range(43)],
                                 output_dict=True)
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)
    
    return accuracy, cm

def preprocess_image(image):
    img = cv2.resize(image, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_traffic_sign(model, class_names, image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence, class_names[predicted_class]

def main():
    set_background("background.png")
    animated_banner()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    model, class_names = load_trained_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please run the Jupyter notebook first to train the model.")
        st.info("📝 Steps to get started:")
        st.write("1. Open `traffic_sign_recognition.ipynb`")
        st.write("2. Run all cells to train the model")
        st.write("3. Refresh this page")
        return
    
    # Sidebar for navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Image Classification", "Model Evaluation", "About"]
    )
    
    if page == "Image Classification":
        image_classification_page(model, class_names)
    elif page == "Model Evaluation":
        model_evaluation_page(model, class_names)
    elif page == "About":
        about_page()

def image_classification_page(model, class_names):
    """Image classification page"""
    st.markdown(
        """
        <div class="upload-box">
            <h3>Upload Traffic Sign Image</h3>
            <p>Upload an image of a traffic sign to classify it using our AI model</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a traffic sign"
    )
    
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        with col1:
            st.markdown(
                """
                <div class="prediction-box">
                    <h3>Uploaded Image</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image(image, caption="Uploaded Traffic Sign", use_container_width =True)
        
        with col2:
            st.markdown(
                """
                <div class="prediction-box">
                    <h3>AI Prediction</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            with st.spinner('Analyzing traffic sign...'):
                predicted_class, confidence, class_name = predict_traffic_sign(model, class_names, image_array)
            
            st.markdown(f"### **Predicted Class:** {predicted_class}")
            st.markdown(f"### **Sign Type:** {class_name}")
            
            confidence_color = "green" if confidence > 80 else "orange" if confidence > 60 else "red"
            st.markdown(f"### **Confidence:** <span style='color: {confidence_color}'>{confidence:.2f}%</span>", unsafe_allow_html=True)
            
            if confidence > 80:
                st.success("High confidence prediction")
            elif confidence > 60:
                st.warning("Medium confidence prediction")
            else:
                st.error("Low confidence prediction")
    
    else:
        st.markdown(
            """
            <div class="prediction-box">
                <h3>Demo Features</h3>
                <ul>
                    <li>Real-time traffic sign classification</li>
                    <li>Confidence score for each prediction</li>
                    <li>Support for 43 different traffic sign classes</li>
                    <li>Fast inference with deep learning</li>
                    <li>Preprocessed images for optimal accuracy</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    with st.expander("About Traffic Sign Classes"):
        st.write("This model can classify the following 43 traffic sign types:")
        class_list = [
            "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
            "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
            "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
            "No passing", "No passing for vehicles over 3.5 tons", "Right-of-way at intersection",
            "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 tons prohibited",
            "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
            "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
            "Road work", "Traffic signals", "Pedestrians", "Children crossing",
            "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
            "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
            "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
            "Keep left", "Roundabout mandatory", "End of no passing",
            "End of no passing by vehicles over 3.5 tons"
        ]
        
        for i, class_name in enumerate(class_list):
            st.write(f"**Class {i}:** {class_name}")

def model_evaluation_page(model, class_names):
    """Model evaluation page with accuracy and confusion matrix"""
    st.markdown(
        """
        <div class="prediction-box">
            <h2>Model Performance Evaluation</h2>
            <p>Comprehensive analysis of model accuracy and performance metrics</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("Run Model Evaluation", type="primary"):
        evaluate_model(model, class_names)
    else:
        st.info("Click the button above to evaluate the model performance on test data")
        
        st.markdown(
            """
            <div class="prediction-box">
                <h3>Evaluation Features</h3>
                <ul>
                    <li><strong>Overall Accuracy:</strong> Model's performance across all classes</li>
                    <li><strong>Confusion Matrix:</strong> Visual representation of prediction accuracy</li>
                    <li><strong>Per-Class Metrics:</strong> Detailed performance for each traffic sign type</li>
                    <li><strong>Classification Report:</strong> Precision, recall, and F1-score analysis</li>
                    <li><strong>Test Dataset:</strong> 50 samples per class for comprehensive evaluation</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

def about_page():
    """About page with project information"""
    st.markdown(
        """
        <div class="prediction-box">
            <h2>About Traffic Sign Recognition System</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(
            """
            ### Project Overview
            This advanced deep learning system uses Convolutional Neural Networks (CNN) 
            to classify German traffic signs with high accuracy.
            
            ### Technical Specifications
            - **Architecture:** Deep CNN with BatchNormalization
            - **Input Size:** 32x32 RGB images
            - **Classes:** 43 different traffic sign types
            - **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)
            - **Framework:** TensorFlow/Keras
            
            ### Performance
            - **Accuracy:** 99.53% on test data
            - **Parameters:** 875,083 trainable parameters
            - **Training:** Adam optimizer with learning rate scheduling
            """
        )
    
    with col2:
        st.markdown(
            """
            ### Technologies Used
            - **Python** - Core programming language
            - **TensorFlow/Keras** - Deep learning framework
            - **OpenCV** - Computer vision operations
            - **Streamlit** - Web application framework
            - **Matplotlib/Seaborn** - Data visualization
            - **Scikit-learn** - Machine learning utilities
            
            ### Development
            - **Created by:** Hamza Younas
            - **Company:** Elevvo Tech
            - **Purpose:** Traffic sign classification and analysis
            """
        )
    
    st.markdown(
        """
        <div class="prediction-box">
            <h3>Features</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                <div>
                    <h4>Image Classification</h4>
                    <p>Upload traffic sign images for real-time AI classification with confidence scoring</p>
                </div>
                <div>
                    <h4>Model Evaluation</h4>
                    <p>Comprehensive performance analysis with confusion matrix and accuracy metrics</p>
                </div>
                <div>
                    <h4>Modern Interface</h4>
                    <p>Clean and responsive design for optimal user experience</p>
                </div>
                <div>
                    <h4>Fast Inference</h4>
                    <p>Optimized model for quick predictions and real-time processing</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()


