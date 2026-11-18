"""pigeonkit: fast moving-object -> pigeon classifier pipeline."""
from .camera import open_camera, Camera, OpenCVCamera, PylonCamera, HarvestersCamera
from .motion import MotionDetector, square_crops, merge_boxes
from .tracker import IoUTracker
from .classifier import OnnxPigeonClassifier, INatBirdClassifier, load_classifier, download_teacher
from .pipeline import PigeonPipeline, Detection, draw

__all__ = [
    "open_camera", "Camera", "OpenCVCamera", "PylonCamera", "HarvestersCamera",
    "MotionDetector", "square_crops", "merge_boxes", "IoUTracker",
    "OnnxPigeonClassifier", "INatBirdClassifier", "load_classifier", "download_teacher",
    "PigeonPipeline", "Detection", "draw",
]
