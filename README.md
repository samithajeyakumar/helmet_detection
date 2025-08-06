# Traffic Violation Detection System Setup Guide

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for version control)

## Step 1: Project Structure
Create the following directory structure:

```
traffic_violation_detection/
├── dashboard.py
├── all_vehicles.py
├── requirements.txt
├── yolo.cfg
├── obj.names
├── models/
│   └── (YOLOv3 weights file will go here)
├── data/
│   └── BB_6d315b1e-956e-43e8-9d62-7d18efed3dd2.mp4
├── utils/
│   ├── __init__.py
│   ├── detection_utils.py
│   └── tracking_utils.py
├── templates/
│   └── index.html
└── static/
    └── css/
        └── style.css
```

## Step 2: Install Dependencies

### Option 1: Using pip directly
```bash
pip install opencv-python==4.7.0.72
pip install numpy>=1.22,<1.24
pip install tensorflow==2.12.0
pip install keras==2.12.0
pip install scikit-learn==1.2.2
pip install matplotlib==3.7.1
pip install pillow==9.5.0
pip install imutils==0.5.4
pip install scipy==1.10.1
pip install flask==2.3.2
```

### Option 2: Using requirements.txt file
```bash
# Navigate to your project directory
cd traffic_violation_detection

# Install all dependencies at once
pip install -r requirements.txt
```

### Option 3: Using virtual environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Download Required Model Files

### YOLOv3 Weights
You need to download the YOLOv3 weights file:

1. **Download YOLOv3 weights** (about 248MB):
   ```bash
   # Create models directory if it doesn't exist
   mkdir models
   
   # Download weights file
   wget https://pjreddie.com/media/files/yolov3.weights -P models/
   ```
   
   Or manually download from: https://pjreddie.com/media/files/yolov3.weights
   
2. **Place the file** in the `models/` directory as `yolov3.weights`

## Step 4: Add Your Video File

1. Place your video file in the `data/` directory
2. Update the video path in both `dashboard.py` and `all_vehicles.py`:
   ```python
   VIDEO_PATH = os.path.join("data", "your_video_file.mp4")
   ```

## Step 5: Create Missing Files

### Create templates/index.html
Copy the HTML content from your files into `templates/index.html`

### Create static/css/style.css
```css
/* Basic styling for the dashboard */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f4f4f4;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background-color: #333;
    color: white;
    text-align: center;
    padding: 1rem;
    margin-bottom: 20px;
}

.main-content {
    display: flex;
    gap: 20px;
}

.video-container {
    flex: 2;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.video-wrapper img {
    width: 100%;
    height: auto;
    border: 2px solid #ddd;
}

.stats-container {
    flex: 1;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.stat-boxes {
    display: grid;
    gap: 10px;
    margin-bottom: 20px;
}

.stat-box {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    text-align: center;
    border-left: 4px solid #007bff;
}

.stat-value {
    font-size: 2em;
    font-weight: bold;
    color: #007bff;
    margin: 5px 0;
}

.chart-container {
    height: 300px;
}

footer {
    text-align: center;
    padding: 20px;
    color: #666;
}
```

## Step 6: Running the Applications

### Option 1: Run the Dashboard (Web Interface)
```bash
# Make sure you're in the project directory
cd traffic_violation_detection

# Run the Flask dashboard
python dashboard.py
```
- Open your web browser and go to `http://localhost:5000`
- The dashboard will show live video processing and statistics

### Option 2: Run Video Processing Only
```bash
# Process video and save output
python all_vehicles.py
```
- This will process the video and display it in a window
- Output video will be saved as `output_video.mp4`
- Press 'q' to quit

## Troubleshooting

### Common Issues:

1. **Module not found errors**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **OpenCV issues**:
   ```bash
   pip uninstall opencv-python
   pip install opencv-python==4.7.0.72
   ```

3. **TensorFlow compatibility**:
   ```bash
   pip install tensorflow==2.12.0 --upgrade
   ```

4. **Video file not found**:
   - Check the video file path in `VIDEO_PATH` variable
   - Ensure the video file exists in the `data/` directory

5. **YOLO weights not found**:
   - Ensure `yolov3.weights` is in the `models/` directory
   - Check file permissions

6. **Port 5000 already in use**:
   ```python
   # Change port in dashboard.py
   app.run(debug=False, host='0.0.0.0', port=5001)
   ```

### Performance Tips:

1. **For better performance**:
   - Use a GPU-enabled version of TensorFlow if you have NVIDIA GPU
   - Reduce video resolution for faster processing
   - Adjust confidence thresholds in detection_utils.py

2. **Memory issues**:
   - Close other applications
   - Reduce batch size or video resolution
   - Use a more powerful machine for real-time processing

## File Descriptions

- **dashboard.py**: Main Flask web application for real-time dashboard
- **all_vehicles.py**: Standalone video processing script
- **requirements.txt**: Python dependencies
- **yolo.cfg**: YOLO model configuration
- **obj.names**: Object class names (COCO dataset)
- **utils/**: Helper modules for detection and tracking
- **templates/**: HTML templates for web interface
- **static/**: CSS and other static files

## Usage Notes

- The system detects vehicles and counts motorcycles as potential helmet violations
- For actual helmet detection, you'd need a custom trained model
- The dashboard updates statistics in real-time
- Video processing can be CPU-intensive for high-resolution videos
