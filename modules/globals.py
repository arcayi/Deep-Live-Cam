import os
from typing import List, Dict

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, 'workflow')

file_types = [
    ('Image', ('*.png','*.jpg','*.jpeg','*.gif','*.bmp')),
    ('Video', ('*.mp4','*.mkv'))
]

source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = None
keep_audio = None
keep_frames = None
many_faces = None

nsfw_filter = None
video_encoder = None
video_quality = None
live_mirror = None
live_resizable = True
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
fp_ui: Dict[str, bool] = {}
camera_input_combobox = None
webcam_preview_running = False
both_faces = None
flip_faces = None
detect_face_right = None
detect_face_right_value = False
show_target_face_box = None
mouth_mask=False
mask_feather_ratio=8
mask_down_size=5
mask_size=11
show_mouth_mask_box=False
flip_faces_value=False
sticky_face_value=0.2
use_pseudo_face = False  # Whether to use pseudo face at all
pseudo_face_threshold = 0.2  # Minimum score to use pseudo face
max_pseudo_face_count = 30  # Maximum number of frames to use pseudo face continuously
face_tracking=False
face_tracking_value=False
target_face1_score=0.00
target_face2_score=0.00