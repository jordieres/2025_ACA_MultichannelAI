import os
import torch

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('Running on device: {}'.format(device))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from tqdm.notebook import tqdm

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import torch
from facenet_pytorch import (MTCNN)

from transformers import (AutoFeatureExtractor,
                          AutoModelForImageClassification,
                          AutoConfig)
                             
from PIL import Image, ImageDraw
from IPython.display import Video
from dataclasses import dataclass
from typing import List, Dict, Literal

# from VideoEmotionAnalizer import VideoEmotionAnalyzer


@dataclass
class VideoEmotionAnalyzer:
    """
    A class for analyzing emotions in videos using deep learning models.
    
    Attributes:
        generate_combined_video (bool): Whether to generate a combined visualization video.
        generate_emotion_plot (bool): Whether to generate an emotion probability plot.
        device (str): The device to run the model on (e.g., "cpu" or "cuda").
        skips (int): One frame each n will be processed.
        method (Literal["mode", "mean", "abs"]): Method to determine the dominant emotion.
    """
    
    generate_combined_video: bool = False
    save_combined_video: bool = False
    generate_emotion_plot: bool = False
    device: str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    skips: int = 3
    method: Literal["mode", "mean", "abs"] = "mode"
    
    def __post_init__(self):
        """
        Initializes the emotion detection models and feature extractor.
        """
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=200,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=False,
            device=self.device
        )
        self.extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
        self.model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
        self.emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    def detect_emotions(self, image: Image.Image):
        """
        Detects emotions in a given image.
        
        Args:
            image (Image.Image): The image containing a face.
        
        Returns:
            tuple: A tuple containing the cropped face and a dictionary of emotion probabilities.
        """
        temporary = image.copy()
        sample = self.mtcnn.detect(temporary)
        if sample[0] is not None:
            box = sample[0][0]
            face = temporary.crop(box)
            inputs = self.extractor(images=face, return_tensors="pt")
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            config = AutoConfig.from_pretrained("trpakov/vit-face-expression")
            id2label = config.id2label
            probabilities = probabilities.detach().numpy().tolist()[0]
            class_probabilities = {id2label[i]: prob for i, prob in enumerate(probabilities)}
            return face, class_probabilities
        return None, None

    def process_video(self, video_frames: List[np.ndarray]):
        """
        Processes a video frame by frame to detect emotions.
        
        Args:
            video_frames (List[np.ndarray]): List of video frames.
            emotions (List[str]): List of possible emotions.
        
        Returns:
            tuple: A tuple containing a list of processed images and emotion probabilities for each frame.
        """
        combined_images, all_class_probabilities = [], []

        for i, frame in enumerate(video_frames):
            combined_image, class_probabilities = self.process_frame(frame)
            if combined_image is not None:
                combined_images.append(combined_image)
            all_class_probabilities.append(class_probabilities)
        
        df = pd.DataFrame(all_class_probabilities) * 100
        return combined_images, df

    def process_frame(self, frame: np.ndarray):
        """
        Processes a video frame to detect emotions.
        
        Args:
            frame (np.ndarray): The video frame.
            emotions (List[str]): List of possible emotions.
        
        Returns:
            tuple: Combined image and class probabilities.
        """
        frame = frame.astype(np.uint8)
        face, class_probabilities = self.detect_emotions(Image.fromarray(frame))
        
        if face is not None:
            combined_image = self.create_combined_image(face, class_probabilities)
        else:
            combined_image = None
            class_probabilities = {emotion: None for emotion in self.emotions}
        
        return combined_image, class_probabilities
    
    def create_combined_image(self, face: Image.Image, class_probabilities: Dict[str, float]):
        """
        Creates an image combining the detected face and a bar plot of emotion probabilities.
        
        Args:
            face (PIL.Image): The detected face.
            class_probabilities (dict): Probabilities of each emotion class.
        
        Returns:
            np.array: The combined image as a numpy array.
        """
        colors = {
            "angry": "red", "disgust": "green", "fear": "gray",
            "happy": "yellow", "neutral": "purple", "sad": "blue", "surprise": "orange"
        }
        palette = [colors[label] for label in class_probabilities.keys()]
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        axs[0].imshow(np.array(face))
        axs[0].axis('off')
        sns.barplot(ax=axs[1], y=list(class_probabilities.keys()), x=[prob * 100 for prob in class_probabilities.values()],
                    hue=list(class_probabilities.keys()), palette=palette, legend=False, orient='h')
        axs[1].set_xlabel('Probability (%)')
        axs[1].set_title('Emotion Probabilities')
        axs[1].set_xlim([0, 100])
        
        canvas = FigureCanvas(fig)
        canvas.draw()

        buffer, (width, height) = canvas.print_to_buffer()
        img = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]  # RGBA format

        plt.close(fig)
        return img

    # def reduce_video_frames(self, video_data: List[np.ndarray]):
    #     """
    #     Reduces the number of frames in the video by skipping frames.
        
    #     Args:
    #         video_data (List[np.ndarray]): List of video frames.
    #         skips (int): Number of frames to skip.
        
    #     Returns:
    #         List[np.ndarray]: A reduced list of video frames.
    #     """
    #     if self.skips < 1:
    #         raise ValueError("The 'skips' value must be at least 1.")
    #     return [video_data[i] for i in range(0, len(video_data), self.skips)]
    def reduce_video_frames(self, video_data: List[np.ndarray]): 
        """
        Reduces the number of frames in the video by keeping only a proportion of frames.
        
        Args:
            video_data (List[np.ndarray]): List of video frames.

        Returns:
            List[np.ndarray]: A reduced list of video frames.
        """
        if not (0 < self.skips <= 1):
            raise ValueError("The 'skips' value must be a proportion between 0 and 1.")

        total_frames = len(video_data)
        num_selected_frames = int(total_frames * self.skips)
        
        if num_selected_frames < 1:
            raise ValueError("The selected proportion is too small, leading to zero frames.")

        # Seleccionar frames distribuidos uniformemente a lo largo del video
        selected_frames = [video_data[i] for i in np.linspace(0, total_frames - 1, num_selected_frames, dtype=int)]
        
        return selected_frames
    
    # def generate_video_from_images(self, combined_images: List[np.ndarray], output_path="/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/VIDEO/combined_images/output_video.mp4", fps=30):
    #     """
    #     Generates a video from a list of images.
        
    #     Args:
    #         combined_images (List[np.ndarray]): List of images to be included in the video.
    #         output_path (str): Output video file name.
    #         fps (int): Frames per second of the original video.
    #         skips (int): Frame reduction factor.
        
    #     Returns:
    #         None (Saves the video to the specified file path).
    #     """
    #     if not combined_images:
    #         raise ValueError("The image list is empty. Cannot generate video.")
        
    #     clip = ImageSequenceClip(combined_images, fps=fps / self.skips)
    #     if self.save_combined_video:
    #         clip.write_videofile(output_path, fps=fps / self.skips, logger=None)
    def generate_video_from_images(self, combined_images: List[np.ndarray], output_path="output_video.mp4", fps=30):
        """
        Generates a video from a list of images.
        
        Args:
            combined_images (List[np.ndarray]): List of images to be included in the video.
            output_path (str): Output video file name.
            fps (int): Frames per second of the original video.
        
        Returns:
            None (Saves the video to the specified file path).
        """
        if not combined_images:
            raise ValueError("The image list is empty. Cannot generate video.")
        
        adjusted_fps = fps * self.skips  # Ajustar FPS para mantener la velocidad relativa
        clip = ImageSequenceClip(combined_images, fps=adjusted_fps)

        if self.save_combined_video:
            clip.write_videofile(output_path, fps=adjusted_fps, logger=None)
    
    def plot_emotion_probabilities(self, df: pd.DataFrame):
        """
        Generates a line graph showing the evolution of emotion probabilities over time.
        
        Args:
            all_class_probabilities (List[Dict[str, float]]): List of dictionaries with emotion probabilities per frame.
        
        Returns:
            None (Displays the emotion probability graph).
        """
        if df.empty:
            raise ValueError("The emotion probabilities list is empty.")
        
        colors = {
            "angry": "red", "disgust": "green", "fear": "gray",
            "happy": "yellow", "neutral": "purple", "sad": "blue", "surprise": "orange"
        }
        
        plt.figure(figsize=(15, 8))
        for emotion in df.columns:
            plt.plot(df[emotion], label=emotion, color=colors.get(emotion, "black"))
        
        plt.xlabel("Frame Order")
        plt.ylabel("Emotion Probability (%)")
        plt.title("Emotion Probabilities Over Time")
        plt.legend()
        plt.show()

    def get_prediction(self, df: pd.DataFrame):
        """
        Determines the dominant emotion from the processed frames based on different methods.
        
        Args:
            df (pd.DataFrame): DataFrame where each row represents a frame and each column represents an emotion probability.
            method (str): Method to determine the dominant emotion. Options are:
                - "mode": Most frequent dominant emotion (default)
                - "highest_mean": Emotion with the highest mean probability
                - "highest_max": Emotion with the highest absolute probability
        
        Returns:
            str: The dominant emotion based on the chosen method.
        """
        if self.method == "mode":
            dominant_emotions = df.idxmax(axis=1)
            prediction = dominant_emotions.mode()[0] 
        elif self.method == "mean":
            prediction = df.mean().idxmax()
        elif self.method == "abs":
            prediction = df.max().idxmax() 
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
        if prediction == 'disgust':
            print('CAMBIO 1')
            prediction = 'fear'
        elif prediction == 'fear':
            print('CAMBIO 2')
            prediction = 'disgust'

        return prediction

    
    def analyze_video(self, video_path: str):
        """
        Analyzes a video to detect emotions and optionally generate visual outputs.
        
        Args:
            video_path (str): The path to the video file.
            skips (int): Number of frames to skip for efficiency.
        
        Returns:
            str: The globally predicted emotion for the video.
        """
        clip = VideoFileClip(video_path)
        vid_fps = clip.fps
        video = clip.without_audio()
        video_data = np.array(list(video.iter_frames()))
        reduced_video = self.reduce_video_frames(video_data)
        combined_images, all_class_probabilities = self.process_video(reduced_video)
        global_prediction = self.get_prediction(all_class_probabilities)
        print("Predicted: ", global_prediction)
        if self.generate_combined_video:
            self.generate_video_from_images(combined_images, "output_video.mp4", vid_fps)
        if self.generate_emotion_plot:
            self.plot_emotion_probabilities(all_class_probabilities)
        return global_prediction
