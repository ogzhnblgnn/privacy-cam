# Real-Time Privacy Preservation Camera 🔒📷

This project is a real-time computer vision application designed to demonstrate privacy preservation techniques using **OpenCV**, **MediaPipe**, and **Face Recognition**.

The application processes live video feed to selectively blur regions based on three distinct privacy stages, utilizing a user-friendly **Tkinter GUI**.

## 🚀 Features

### 1. Privacy Modes
* **Stage 1 (Face Focus):** Detects faces and keeps them sharp while blurring the rest of the background.
* **Stage 2 (Portrait Mode):** Segments the full human body (Selfie Segmentation) and blurs the background.
* **Stage 3 (Selective Privacy):** Identifies a specific "Target Person" using face recognition. The target person remains clear, while all other individuals and the background are blurred.

### 2. Customization
* **Blur Methods:** Choose between **Gaussian Blur** (Soft) and **Pixelation** (Mosaic/Censored style) for comparison.
* **Target Registration:** One-click registration for the target person in Stage 3.

## 🛠️ Technologies Used

* **Python 3.10**
* **OpenCV:** For basic image processing operations.
* **MediaPipe:**
    * *Face Detection:* For Stage 1.
    * *Selfie Segmentation:* For Stage 2 (Body segmentation).
* **Face Recognition (dlib):** For encoding and matching faces in Stage 3.
* **Tkinter:** For the graphical user interface (GUI).

## ⚙️ Installation

Since this project relies on `dlib` and `mediapipe`, it is recommended to use **Conda** to manage dependencies and avoid compilation errors.

**1. Clone the repository**
```bash
git clone https://github.com/ogzhnblgnn/privacy-preservation-camera.git
cd privacy-preservation-camera
```

**2. Create a Conda Environment (Python 3.10 is required)**
```bash
conda create -n privacy_cam python=3.10 -y
conda activate privacy_cam
```

**3. Install Dependencies**
First, install `dlib` and `cmake` from conda-forge to handle C++ dependencies:
```bash
conda install -c conda-forge dlib cmake -y
```

Then, install the remaining Python packages:
```bash
pip install opencv-python mediapipe face-recognition numpy pillow
```

## ▶️ Usage

Run the main script to start the application:

```bash
python main.py
```

### How to Use the Interface:
1.  **Normal:** Shows the raw camera feed.
2.  **Blur Method:** Select "Gaussian" or "Pixelate" from the dropdown.
3.  **Stage 1 & 2:** Click the buttons to activate respective privacy modes.
4.  **Stage 3 (Target Mode):**
    * First, position yourself in front of the camera.
    * Click **"Hedef Kişiyi Kaydet" (Register Target)**. The button will turn green.
    * Click **"Aşama 3"** to activate. Now, only you will be visible; intruders will be blurred!

## 🧩 How It Works (Logic)

* **Face Detection:** Uses MediaPipe's lightweight model to find bounding boxes of faces. We create a mask where the face ROI (Region of Interest) is white and the rest is black, then blend the original and blurred frames.
* **Body Segmentation:** Uses MediaPipe's Selfie Segmentation model to generate a binary mask separating the "person" from the "background".
* **Face Recognition:**
    * The system computes a **128-dimensional face encoding vector** for the registered user.
    * In every frame, it calculates the **Euclidean distance** between the detected faces and the registered vector.
    * If the distance is below a threshold (0.5), the face is classified as the "Target" and preserved.


## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
