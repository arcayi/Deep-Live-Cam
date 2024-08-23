from typing import Any, List
import cv2
import insightface
import threading
import numpy as np
import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, get_one_face_left, get_one_face_right,get_face_analyser
from modules.typing import Face, Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-SWAPPER'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(modules.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128_fp16.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=modules.globals.execution_providers)
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def create_mouth_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
      
    mouth_cutout = None
    
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Get mouth landmarks (indices 52 to 71 typically represent the outer mouth)
        mouth_points = landmarks[52:71].astype(np.int32)
        
        # Add padding to mouth area
        min_x, min_y = np.min(mouth_points, axis=0)
        max_x, max_y = np.max(mouth_points, axis=0)
        min_x = max(0, min_x - (15*6))
        min_y = max(0, min_y - 10)
        max_x = min(frame.shape[1], max_x + (15*6))
        max_y = min(frame.shape[0], max_y + (90*6))
        
        # Create a polygon mask for the mouth
        mouth_polygon = np.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
        cv2.fillPoly(mask, [mouth_polygon], 255)
        
        # Extract the mouth area from the frame using the calculated bounding box
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y)



def create_feathered_mask(shape, feather_amount=30):
    mask = np.zeros(shape[:2], dtype=np.float32)
    center = (shape[1] // 2, shape[0] // 2)
    cv2.ellipse(mask, center, (shape[1] // 2 - feather_amount, shape[0] // 2 - feather_amount), 
                0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)
    return mask / np.max(mask)

def apply_mouth_area(frame: np.ndarray, mouth_cutout: np.ndarray, mouth_box: tuple) -> np.ndarray:
    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y
    

    # Resize the mouth cutout to match the mouth box size
    if mouth_cutout is None or box_width is None or box_height is None:
        return
    resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
    
    # Extract the region of interest (ROI) from the target frame
    roi = frame[min_y:max_y, min_x:max_x]
    
    # Ensure the ROI and resized_mouth_cutout have the same shape
    if roi.shape != resized_mouth_cutout.shape:
        resized_mouth_cutout = cv2.resize(resized_mouth_cutout, (roi.shape[1], roi.shape[0]))
    
    # Apply color transfer from ROI to mouth cutout
    color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)
    
    # Create a feathered mask with increased feather amount
    feather_amount = min(30, box_width // 15, box_height // 15)
    mask = create_feathered_mask(resized_mouth_cutout.shape, feather_amount)
    
    # Blend the color-corrected mouth cutout with the ROI using the feathered mask
    mask = mask[:,:,np.newaxis]  # Add channel dimension to mask
    blended = (color_corrected_mouth * mask + roi * (1 - mask)).astype(np.uint8)
    
    # Place the blended result back into the frame
    frame[min_y:max_y, min_x:max_x] = blended
    
    return frame

def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    # Reshape mean and std to be broadcastable
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Perform the color transfer
    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)


def process_frame(source_face: List[Face], temp_frame: Frame) -> Frame:

    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                if modules.globals.flip_faces:
                    # Get the mouth mask, cutout, and bounding box
                    if modules.globals.mouth_mask is True:
                        mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_face, temp_frame)

                    temp_frame = swap_face(source_face[1], target_face, temp_frame)
                    
                    if modules.globals.mouth_mask is True:
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box) 
                else:
                    if modules.globals.mouth_mask is True:
                        # Get the mouth mask, cutout, and bounding box
                        mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_face, temp_frame)

                    temp_frame = swap_face(source_face[0], target_face, temp_frame)
                    
                    if modules.globals.mouth_mask is True:
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box) 
    else:
        target_faces = get_two_faces(temp_frame)
        # Check if more then one target face is found
        if len(target_faces) >= 2:
            # Swap both target faces when with source image. Works best when source image 
            # has two faces. If source image has one face then one face is used for both 
            # target faces 
            if modules.globals.both_faces:
                # Flip source faces left to right
                if modules.globals.flip_faces:
                    # Swap right source face with left target face
                    # Get the mouth mask, cutout, and bounding box    
                    if modules.globals.mouth_mask is True:
                        mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_faces[0], temp_frame)

                    temp_frame = swap_face(source_face[1], target_faces[0], temp_frame)

                    # Apply the mouth cutout
                    if modules.globals.mouth_mask is True:
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box) 
                    
                    if modules.globals.mouth_mask is True:
                        # Get the mouth mask, cutout, and bounding box
                        mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_faces[1], temp_frame)

                    # Swap left source face with right target face
                    temp_frame = swap_face(source_face[0], target_faces[1], temp_frame)

                    if modules.globals.mouth_mask is True:
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)
                else:
                    if modules.globals.mouth_mask is True:
                        # Get the mouth mask, cutout, and bounding box
                        mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_faces[0], temp_frame) 

                    # Swap left source face with left target face
                    temp_frame = swap_face(source_face[0], target_faces[0], temp_frame)
                    
                    if modules.globals.mouth_mask is True:
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box) 

                    if modules.globals.mouth_mask is True:    
                        # Get the mouth mask, cutout, and bounding box
                        mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_faces[1], temp_frame)

                    # Swap right source face with right target face
                    temp_frame = swap_face(source_face[1], target_faces[1], temp_frame)
                    
                    if modules.globals.mouth_mask is True: 
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)

            # When we have two target faces we can replace left or right face
            # Swap one face with left target face or right target face
            elif modules.globals.detect_face_right:
                # Swap left source face with right target face
                # Get the mouth mask, cutout, and bounding box
                if modules.globals.mouth_mask is True:
                    mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_faces[1], temp_frame)

                if modules.globals.flip_faces:
                    # Swap right source face with right target face
                    temp_frame = swap_face(source_face[1], target_faces[1], temp_frame)
                    
                    if modules.globals.mouth_mask is True:
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)
                else:
                    # Swap left source face with right target face
                    temp_frame = swap_face(source_face[0], target_faces[1], temp_frame)
                    
                    if modules.globals.mouth_mask is True:
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)

            else:
                if modules.globals.mouth_mask is True:
                    # Get the mouth mask, cutout, and bounding box
                    mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_faces[0], temp_frame)

                # Swap left source face with left target face
                if modules.globals.flip_faces:
                    # Swap left source face with left target face
                    temp_frame = swap_face(source_face[1], target_faces[0], temp_frame)
                    
                    if modules.globals.mouth_mask is True:
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)
                else:
                    
                    # Swap right source face with left target face
                    temp_frame = swap_face(source_face[0], target_faces[0], temp_frame)
                    
                    if modules.globals.mouth_mask is True:
                        # Apply the mouth cutout
                        temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)

        elif len(target_faces) == 1:
            # If only one target face is found, swap with the first source face
            # Swap left source face with left target face
            
            if modules.globals.mouth_mask is True:
                # Get the mouth mask, cutout, and bounding box
                mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(target_faces[0], temp_frame)

            if modules.globals.flip_faces:
                # Swap left source face with left target face
                temp_frame = swap_face(source_face[1], target_faces[0], temp_frame)
                
                if modules.globals.mouth_mask is True:
                    # Apply the mouth cutout
                    temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)
            else:

                # Swap right source face with left target face
                temp_frame = swap_face(source_face[0], target_faces[0], temp_frame)
                
                if modules.globals.mouth_mask is True:
                    # Apply the mouth cutout
                    temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)

    
    if modules.globals.show_target_face_box:
        if modules.globals.many_faces:
            temp_frame = get_face_analyser().draw_on(temp_frame, many_faces)
        else:
            temp_frame = get_face_analyser().draw_on(temp_frame, target_faces)



    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    
    source_image_left = None  # Initialize variable for the selected face image
    source_image_right = None  # Initialize variable for the selected face image

    if source_image_left is None and source_path:
        source_image_left = get_one_face_left(cv2.imread(source_path))
    if source_image_right is None and source_path:
        source_image_right = get_one_face_right(cv2.imread(source_path))


    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        try:
            result = process_frame([source_image_left,source_image_right], temp_frame)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(exception)
            pass
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    
    source_image_left = None  # Initialize variable for the selected face image
    source_image_right = None  # Initialize variable for the selected face image

    if source_image_left is None and source_path:
        source_image_left = get_one_face_left(cv2.imread(source_path))
    if source_image_right is None and source_path:
        source_image_right = get_one_face_right(cv2.imread(source_path))

    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    result = process_frame([source_image_left,source_image_right], target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

def get_two_faces(frame: Frame) -> List[Face]:
    faces = get_many_faces(frame)
    if faces:
        # Sort faces from left to right based on the x-coordinate of the bounding box
        sorted_faces = sorted(faces, key=lambda x: x.bbox[0])
        return sorted_faces[:2]  # Return up to two faces, leftmost and rightmost
    return []