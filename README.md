# Anomaly Detection System

A **real-time surveillance system** built using **YOLOv8**, **OpenCV**, and **Streamlit**, capable of detecting unusual activities such as **running** and **loitering** in video feeds.  

This project combines deep learning-based object detection with motion analysis to flag suspicious human activity from uploaded or live video streams.

---

## [View live](https://anomaly-detection-from-cctv.streamlit.app/)

## Demo




## Features

- 🎥 Process uploaded videos or use a **live webcam** feed  
- ⚙️ Adjustable detection thresholds via sidebar  
- 🧍 Detect **running** or **loitering** anomalies  
- 💾 Export and view processed output videos directly in Streamlit  
- 🌐 One-click deployment on [Streamlit Cloud](https://streamlit.io/cloud)

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| Interface | [Streamlit](https://streamlit.io/) |
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Video Processing | OpenCV (headless) |
| Backend | Python 3.10+ |
| Deployment | Streamlit Cloud |

---

## ⚙️ Installation & Setup (Local)

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/anomaly-detection.git
   cd anomaly-detection
   ```
   
2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Detection**
   Upload an existing video or use webcam






   

