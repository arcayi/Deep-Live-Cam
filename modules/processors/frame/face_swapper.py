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
    conditional_download(download_directory_path, ['https://huggingface.co/ivideogameboss/iroopdeepfacecam/blob/main/inswapper_128_fp16.onnx'])
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

def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convert landmarks to int32
        landmarks = landmarks.astype(np.int32)
        
        # Extract facial features
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calculate forehead extension
        right_eyebrow_top = np.min(right_eye_brow[:, 1])
        left_eyebrow_top = np.min(left_eye_brow[:, 1])
        eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)
        
        face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
        forehead_height = face_top - eyebrow_top
        extended_forehead_height = int(forehead_height * 5.0)  # Extend by 50%

        # Create forehead points
        forehead_left = right_side_face[0].copy()
        forehead_right = left_side_face[-1].copy()
        forehead_left[1] -= extended_forehead_height
        forehead_right[1] -= extended_forehead_height

        # Combine all points to create the face outline
        face_outline = np.vstack([
            [forehead_left],
            right_side_face,
            left_side_face[::-1],  # Reverse left side to create a continuous outline
            [forehead_right]
        ])

        # Calculate padding
        padding = int(np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05)  # 5% of face width

        # Create a slightly larger convex hull for padding
        hull = cv2.convexHull(face_outline)
        hull_padded = []
        for point in hull:
            x, y = point[0]
            center = np.mean(face_outline, axis=0)
            direction = np.array([x, y]) - center
            direction = direction / np.linalg.norm(direction)
            padded_point = np.array([x, y]) + direction * padding
            hull_padded.append(padded_point)

        hull_padded = np.array(hull_padded, dtype=np.int32)

        # Fill the padded convex hull
        cv2.fillConvexPoly(mask, hull_padded, 255)

        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask

def blur_edges(mask: np.ndarray, blur_amount: int = 40) -> np.ndarray:
    blur_amount = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
    return cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()
    
    # Apply the face swap
    swapped_frame = face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
    
    # Create a mask for the target face
    target_mask = create_face_mask(target_face, temp_frame)
    
    # Blur the edges of the mask
    blurred_mask = blur_edges(target_mask)
    blurred_mask = blurred_mask / 255.0
    
    # Ensure the mask has 3 channels to match the frame
    blurred_mask_3channel = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)
    
    # Blend the swapped face with the original frame using the blurred mask
    blended_frame = (swapped_frame * blurred_mask_3channel + 
                     temp_frame * (1 - blurred_mask_3channel))
    
    return blended_frame.astype(np.uint8)

def create_mouth_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Get relevant landmarks
        nose_tip = landmarks[80].astype(np.float32)
        center_bottom = landmarks[73].astype(np.float32)
        left_eye = landmarks[88].astype(np.float32)
        right_eye = landmarks[38].astype(np.float32)
        
        # Calculate the vectors
        center_to_nose = nose_tip - center_bottom
        center_to_left_eye = nose_tip - left_eye
        center_to_right_eye = nose_tip - right_eye
        
        # Calculate the lengths of the vectors
        center_to_nose_length = np.linalg.norm(center_to_nose)
        center_to_left_eye_length = np.linalg.norm(center_to_left_eye)
        center_to_right_eye_length = np.linalg.norm(center_to_right_eye)
        
        # Find the largest length
        largest_length = max(center_to_nose_length, center_to_left_eye_length, center_to_right_eye_length)
        
        # Determine which vector is the largest
        if largest_length == center_to_nose_length:
            largest_vector = center_to_nose
        elif largest_length == center_to_left_eye_length:
            largest_vector = center_to_left_eye
        else:
            largest_vector = center_to_right_eye
        
        # Set mask height using modules.globals.mask_size
        base_height = largest_vector * 0.8
        mask_height = np.linalg.norm(base_height) * modules.globals.mask_size 
        
        # Calculate mask top (pushed further down from nose tip) and bottom
        mask_top = nose_tip + center_to_nose * 0.2  # Increased from 0.1 to 0.3
        mask_bottom = mask_top + center_to_nose * (mask_height / np.linalg.norm(center_to_nose))
        
        # Calculate horizontal range
        mouth_points = landmarks[52:71].astype(np.float32)
        mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
        
        # Set mask width using modules.globals.mask_size
        base_width = mouth_width * 0.4
        mask_width = base_width * modules.globals.mask_size
        
        # Calculate mask center
        mask_center = (mask_top + mask_bottom) / 2
        
        # Calculate mask left and right edges
        mask_direction = np.array([-center_to_nose[1], center_to_nose[0]])  # Perpendicular to chin-to-nose
        mask_direction /= np.linalg.norm(mask_direction)
        
        # Create mask polygon
        mask_polygon = np.array([
            mask_top + mask_direction * (mask_width / 2),
            mask_top - mask_direction * (mask_width / 2),
            mask_bottom - mask_direction * (mask_width / 2),
            mask_bottom + mask_direction * (mask_width / 2)
        ]).astype(np.int32)
        
        # Ensure the mask stays within the frame
        mask_polygon[:, 0] = np.clip(mask_polygon[:, 0], 0, frame.shape[1] - 1)
        mask_polygon[:, 1] = np.clip(mask_polygon[:, 1], 0, frame.shape[0] - 1)
        
        # Draw the mask
        cv2.fillPoly(mask, [mask_polygon], 255)
        
        # Calculate bounding box for the mouth cutout
        min_x, min_y = np.min(mask_polygon, axis=0)
        max_x, max_y = np.max(mask_polygon, axis=0)
        
        # Extract the masked area from the frame
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
    try:
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
    except Exception:
        pass

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
    # Pre-compute face detection
    all_target_faces = get_many_faces(temp_frame)
    
    # Sort faces from left to right based on their x-coordinate
    all_target_faces.sort(key=lambda face: face.bbox[0])

    # Determine which faces to process based on settings
    if modules.globals.many_faces:
        target_faces = all_target_faces
    elif modules.globals.detect_face_right:
        target_faces = all_target_faces[-2:][::-1]  # Last two faces, reversed
    else:
        target_faces = all_target_faces[:2]  # First two faces

    # Pre-compute mouth masks if needed
    mouth_masks = []
    if modules.globals.mouth_mask:
        if modules.globals.many_faces:
            for face in target_faces:
                mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(face, temp_frame)
                mouth_masks.append((mouth_mask, mouth_cutout, mouth_box))
        elif modules.globals.both_faces:
            for face in target_faces[:2]:
                mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(face, temp_frame)
                mouth_masks.append((mouth_mask, mouth_cutout, mouth_box))
        else:
            face = target_faces[0] if target_faces else None
            if face:
                mouth_mask, mouth_cutout, mouth_box = create_mouth_mask(face, temp_frame)
                mouth_masks.append((mouth_mask, mouth_cutout, mouth_box))

    # Determine the active source face index
    active_source_index = 1 if modules.globals.flip_faces else 0

    # Process faces
    if modules.globals.many_faces:
        for i, target_face in reversed(list(enumerate(target_faces))):
            if modules.globals.both_faces:
                source_index = (len(target_faces) - 1 - i) % len(source_face)  # Alternate between source faces in reverse
            else:
                source_index = active_source_index  # Use the active source face for all targets
            
            temp_frame = swap_face(source_face[source_index], target_face, temp_frame)
            if modules.globals.mouth_mask:
                mouth_mask, mouth_cutout, mouth_box = mouth_masks[len(target_faces) - 1 - i]
                temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)
    else:
        faces_to_process = 2 if modules.globals.both_faces else 1
        for i in range(min(faces_to_process, len(target_faces))):
            reverse_i = len(target_faces) - 1 - i
            source_index = reverse_i if modules.globals.flip_faces else (1 - reverse_i)
            temp_frame = swap_face(source_face[source_index], target_faces[reverse_i], temp_frame)
            if modules.globals.mouth_mask and reverse_i < len(mouth_masks):
                mouth_mask, mouth_cutout, mouth_box = mouth_masks[reverse_i]
                temp_frame = apply_mouth_area(temp_frame, mouth_cutout, mouth_box)

    # Draw face boxes if needed
    if modules.globals.show_target_face_box:
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