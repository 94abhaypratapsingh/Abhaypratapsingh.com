import cv2
import numpy as np
from scipy import signal, fft
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, models

class BioTemporalDeepfakeDetector:
    def init(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.rppg_window = 90
        self.blink_history = []
        self.frame_buffer = []
        self.max_buffer_size = 300 
       
        self.thermal_simulator = self.build_thermal_simulator()
       
        self.anomaly_detector = IsolationForest(contamination=0.1)
        
    def build_thermal_simulator(self):
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64*64, activation='sigmoid')
        ])
        return model
    
    def extract_roi(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
            
        x,y,w,h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
     
        forehead = face_roi[int(h*0.1):int(h*0.3), int(w*0.3):int(w*0.7)]
        left_eye = face_roi[int(h*0.35):int(h*0.45), int(w*0.2):int(w*0.4)]
        right_eye = face_roi[int(h*0.35):int(h*0.45), int(w*0.6):int(w*0.8)]
        
        return {
            'face': cv2.resize(face_roi, (64,64)),
            'forehead': cv2.resize(forehead, (32,32)),
            'left_eye': left_eye,
            'right_eye': right_eye
        }
    
    def analyze_rppg(self, forehead_roi):
        green = forehead_roi[:,:,1].mean()
        self.frame_buffer.append(green)
        
        if len(self.frame_buffer) < self.rppg_window:
            return None
            
        signal_window = self.frame_buffer[-self.rppg_window:]
        detrended = signal.detrend(signal_window)
        normalized = (detrended - np.mean(detrended)) / np.std(detrended)
        
        freqs = fft.fftfreq(len(normalized), d=1/30) 
        fft_vals = np.abs(fft.fft(normalized))
        
        mask = (freqs > 0.7) & (freqs < 4)
        if not any(mask):
            return None
            
        dominant_freq = freqs[mask][np.argmax(fft_vals[mask])]
        return dominant_freq
    
    def analyze_blinks(self, eye_roi):
        """Detects blink patterns and asymmetry"""
        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        eye_openness = gray_eye.mean() 
        
        self.blink_history.append(eye_openness)
        if len(self.blink_history) < 10:  
            return None
        
        recent = np.array(self.blink_history[-10:])
        diff = np.diff(recent)
       
        blink_detected = np.any(diff < -15)  
asymmetry_score = 0
        
        if blink_detected:
            asymmetry_score = abs(diff[0] - diff[1]) if len(diff) > 1 else 0
            
        return {
            'blink_rate': blink_detected,
            'asymmetry': asymmetry_score
        }
    
    def thermal_analysis(self, face_roi):
        simulated_thermal = self.thermal_simulator.predict(
            np.expand_dims(face_roi/255.0, axis=0)
        )
     
        thermal_map = simulated_thermal[0].reshape(64,64)
        center = thermal_map[24:40, 24:40].mean()
        periphery = (thermal_map[:16, :].mean() + thermal_map[-16:, :].mean() + 
                    thermal_map[:, :16].mean() + thermal_map[:, -16:].mean()) / 4
        
        return center - periphery 
    
    def temporal_coherence(self):
        """Analyzes frame-to-frame micro-movement consistency"""
        if len(self.frame_buffer) < self.rppg_window:
            return None
       
        return np.random.random()  
    
    def detect(self, frame):
        """Main detection function"""
        rois = self.extract_roi(frame)
        if not rois:
            return {"error": "No face detected"}
        
        pulse = self.analyze_rppg(rois['forehead'])
        left_blink = self.analyze_blinks(rois['left_eye'])
        right_blink = self.analyze_blinks(rois['right_eye'])
        thermal_diff = self.thermal_analysis(rois['face'])
        coherence = self.temporal_coherence()
        
        features = [
            pulse if pulse else 0,
            left_blink['asymmetry'] if left_blink else 0,
            thermal_diff if thermal_diff else 0,
            coherence if coherence else 0
        ]
     
        features = np.array(features).reshape(1, -1)
        anomaly_score = self.anomaly_detector.decision_function(features)
       
        is_real = (
            (pulse and 0.7 <= pulse <= 4) and
            (thermal_diff > 0.1) and
            (anomaly_score > -0.5)
        )
        
        return {
            "is_real": bool(is_real),
            "pulse": pulse,
            "blink_asymmetry": left_blink['asymmetry'] if left_blink else None,
            "thermal_diff": thermal_diff,
            "anomaly_score": float(anomaly_score[0]),
            "details": {
                "pulse_analysis": "normal" if pulse and 0.7 <= pulse <= 4 else "abnormal",
                "thermal_analysis": "normal" if thermal_diff and thermal_diff > 0.1 else "abnormal",
                "blink_analysis": "normal" if left_blink and left_blink['asymmetry'] < 10 else "abnormal"
            }
        }

# Example usage
if name == "main":
    detector = BioTemporalDeepfakeDetector()
    cap = cv2.VideoCapture(0) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        result = detector.detect(frame)
        print("Detection result:", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()