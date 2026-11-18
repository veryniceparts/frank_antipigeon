"""pigeonkit: fast moving-object -> pigeon classifier pipeline."""
from .camera import open_camera, Camera, OpenCVCamera, GstCamera, AravisCamera, PylonCamera, HarvestersCamera, csi_pipeline
from .motion import MotionDetector, square_crops, merge_boxes
from .tracker import IoUTracker
from .classifier import (OnnxPigeonClassifier, INatBirdClassifier, TensorRTPigeonClassifier,
                         load_classifier, download_teacher)
from .pipeline import PigeonPipeline, Detection, draw

__all__ = [
    "open_camera", "Camera", "OpenCVCamera", "GstCamera", "AravisCamera", "csi_pipeline", "PylonCamera", "HarvestersCamera",
    "MotionDetector", "square_crops", "merge_boxes", "IoUTracker",
    "OnnxPigeonClassifier", "INatBirdClassifier", "TensorRTPigeonClassifier", "load_classifier", "download_teacher",
    "PigeonPipeline", "Detection", "draw",
]
