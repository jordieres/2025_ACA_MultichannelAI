import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from abc import ABC, abstractmethod
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
from moviepy.video.io.VideoFileClip import VideoFileClip
from fer import FER
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from dataclasses import dataclass, field
from typing import List



@dataclass
class FaceDetector:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn: MTCNN = field(default_factory=lambda: MTCNN(keep_all=False, post_process=True, device='cuda' if torch.cuda.is_available() else 'cpu'))

    def detect_faces(self, image: Image.Image):
        sample = self.mtcnn.detect(image)
        if sample[0] is not None:
            box = sample[0][0]
            face = image.crop(box)
            return face
        return None

    def recognize_faces(self, frame: np.ndarray) -> List[np.array]:
        """
        Detects faces in the given image and returns the facial images cropped from the original.
        """
        def detect_face(frame: np.ndarray):
            bounding_boxes, probs = self.mtcnn.detect(frame, landmarks=False)
            if probs is None or probs[0] == None:
                return []
            bounding_boxes = bounding_boxes[probs > 0.9]
            return bounding_boxes

        bounding_boxes = detect_face(frame)
        facial_images = []
        for bbox in bounding_boxes:
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            facial_images.append(frame[y1:y2, x1:x2, :])
        return facial_images



class EmotionRecognizer(ABC):
    """Abstract class for emotion recognition models."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    @abstractmethod
    def predict_emotion(self, face: Image.Image):
        pass



@dataclass
class VITRecognizer(EmotionRecognizer):
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor: AutoFeatureExtractor = field(default_factory=lambda: AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression"))
    model: AutoModelForImageClassification = field(default_factory=lambda: AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").to('cuda' if torch.cuda.is_available() else 'cpu'))
    config: AutoConfig = field(default_factory=lambda: AutoConfig.from_pretrained("trpakov/vit-face-expression"))
    id2label: dict = field(init=False)

    def __post_init__(self):
        self.id2label = self.config.id2label

    def predict_emotion(self, face: Image.Image):
        inputs = self.extractor(images=face, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probabilities = probabilities.detach().cpu().numpy().tolist()[0]
        class_probabilities = {self.id2label[i]: prob for i, prob in enumerate(probabilities)}
        return class_probabilities

@dataclass
class FERRecognizer(EmotionRecognizer):
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fer_detector: FER = field(default_factory=lambda: FER(mtcnn=True))
    
    def predict_emotion(self, face: Image.Image):
        img_cv = np.array(face)[:, :, ::-1]  # Convert PIL to OpenCV
        detections = self.fer_detector.detect_emotions(img_cv)
        return detections[0]['emotions'] if detections else None

@dataclass
class EmotiEffRecognizer(EmotionRecognizer):
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model: str = "enet_b0_8_best_afew"
    emotion_mapping: dict = field(default_factory=lambda: {"anger": "angry", "happiness": "happy", "sadness": "sad", "fear": "fear", "surprise": "surprise", "disgust": "disgust", "neutral": "neutral"})
    emotion_labels: list = field(default_factory=lambda: ["angry", "happy", "sad", "fear", "surprise", "disgust", "neutral"])

    def __post_init__(self):
        self.recognizer = EmotiEffLibRecognizer(engine="torch", model_name=self.model, device=self.device)

    def map_emotion(self, emotion: str) -> str:
        return self.emotion_mapping.get(emotion, emotion)

    def predict_emotion(self, facial_images):
        # facial_images = face_detector.recognize_faces(frame)
        if not facial_images:
            return {emotion: 0.0 for emotion in self.emotion_labels}
        
        _, scores = self.recognizer.predict_emotions(facial_images[0])
        softmax_scores = torch.nn.functional.softmax(torch.tensor(scores), dim=1)
        emotion_probs = {self.emotion_labels[i]: float(softmax_scores[0, i]) for i in range(len(self.emotion_labels))}
    
        return emotion_probs


@dataclass
class VideoProcessor:
    skips: float = 0.1  # Proporción de frames a mantener (entre 0 y 1)
    
    def extract_frames(self, video_path: str):
        clip = VideoFileClip(video_path)
        video = clip.without_audio()
        video_data = np.array(list(video.iter_frames()))
        return self.reduce_video_frames(video_data)

    def reduce_video_frames(self, video_data: list):
        if not (0 < self.skips <= 1):
            raise ValueError("The 'skips' value must be a proportion between 0 and 1.")
        
        total_frames = len(video_data)
        num_selected_frames = int(total_frames * self.skips)
        
        if num_selected_frames < 1:
            raise ValueError("The selected proportion is too small, leading to zero frames.")
        
        return [video_data[i] for i in np.linspace(0, total_frames - 1, num_selected_frames, dtype=int)]

@dataclass
class EmotionVideoAnalyzer:
    recognizer: EmotionRecognizer
    face_detector: FaceDetector
    processor: VideoProcessor
    
    def analyze_video(self, video_path: str):
        frames = self.processor.extract_frames(video_path)
        predictions = []

        if isinstance(self.recognizer, EmotiEffRecognizer):
            for frame in frames:
                facial_images = self.face_detector.recognize_faces(frame)
                emotion = self.recognizer.predict_emotion(facial_images)
                if emotion is not None:
                    predictions.append(emotion)
            return pd.DataFrame(predictions)
        
        if isinstance(self.recognizer, FERRecognizer) or isinstance(self.recognizer, VITRecognizer):
            for frame in frames:
                face = self.face_detector.detect_faces(Image.fromarray(frame))
                if face:
                    emotion = self.recognizer.predict_emotion(face)
                    if emotion is not None:
                        predictions.append(emotion)
            return pd.DataFrame(predictions)
        


@dataclass
class VideoEmotionAnalysis:
    mode: str
    skips: float = 0.1
    method: str = "mode"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotieff_model: str = "enet_b0_8_best_afew"
    
    def __post_init__(self):
        self.face_detector = FaceDetector(self.device)
        self.video_processor = VideoProcessor(self.skips)
        
        match self.mode:
            case "vit":
                self.recognizer = VITRecognizer(self.device)
            case "fer":
                self.recognizer = FERRecognizer(self.device)
            case "emotieff":
                self.recognizer = EmotiEffRecognizer(self.device, model=self.emotieff_model)
            case _:
                raise ValueError("Unsupported mode. Choose 'vit', 'fer', or 'emotieff'.")
        
        self.analyzer = EmotionVideoAnalyzer(self.recognizer, self.face_detector, self.video_processor)
    
    def analyze_video(self, video_path: str) -> str:
        df = self.analyzer.analyze_video(video_path)
        if self.mode != "emotieff":
            pred = self.change_disgust_fear(self.get_prediction(df)) if not df.empty else 'unknown'
        else:
            pred = self.get_prediction(df) if not df.empty else 'unknown'
        print(f"Predicted Emotion: {pred}")
        return pred 
    
    def get_prediction(self, df: pd.DataFrame):
        match self.method:
            case "mode":
                return df.idxmax(axis=1).mode()[0] 
            case "mean":
                return df.mean().idxmax()
            case "abs":
                return df.max().idxmax() 
            case _:
                raise ValueError(f"Unknown method: {self.method}")
            
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Path" not in df.columns:
            raise ValueError("DataFrame must contain a 'Path' column with video file paths.")
        
        df["classification"] = df["Path"].apply(self.analyze_video)
        return df
    
    def change_disgust_fear(self, prediction: str) -> str:
        if self.mode in ["vit", "fer", "emotieff"]:
            if prediction == "disgust":
                return "fear"
            elif prediction == "fear":
                return "disgust"
        return prediction