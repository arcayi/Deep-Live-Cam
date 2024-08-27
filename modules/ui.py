import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps
import numpy as np
import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face, get_one_face_left, get_one_face_right
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video, resolve_relative_path, has_image_extension

global camera
camera = None

ROOT = None
ROOT_HEIGHT = 750
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH  = 1200
PREVIEW_DEFAULT_WIDTH  = 640
PREVIEW_DEFAULT_HEIGHT = 360
BLUR_SIZE=1

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None

img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, preview_size_var, mouth_mask_var,mask_size_var,mask_down_size_var,mask_feather_ratio_var

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    # Image preview area
    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.05, rely=0.05, relwidth=0.4, relheight=0.3)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.55, rely=0.05, relwidth=0.4, relheight=0.3)

    # Buttons for selecting source and target
    select_face_button = ctk.CTkButton(root, text='Select a face/s\n(left face)(right face)', cursor='hand2', command=lambda: select_source_path())
    select_face_button.place(relx=0.05, rely=0.33, relwidth=0.4, relheight=0.08)

    select_target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2', command=lambda: select_target_path())
    select_target_button.place(relx=0.55, rely=0.33, relwidth=0.4, relheight=0.08)

    # Left column of switches
    y_start = 0.43
    y_increment = 0.06

    switches_left = [
        ('Show both faces', 'both_faces'),
        ('Flip left/right faces', 'flip_faces'),
        ('Detect face from right', 'detect_face_right'),
        ('Many faces', 'many_faces'),
        ('Show target face box','show_target_face_box'),
        ('Show mouth mask box','show_mouth_mask_box')
    ]

    for i, (text, attr) in enumerate(switches_left):
        value = ctk.BooleanVar(value=getattr(modules.globals, attr, False))
        switch = ctk.CTkSwitch(root, text=text, variable=value, cursor='hand2',
                               command=lambda a=attr, v=value: setattr(modules.globals, a, v.get()))
        switch.place(relx=0.05, rely=y_start + i*y_increment, relwidth=0.4)



    # Right column of switches
    switches_right = [
        ('Mirror webcam', 'live_mirror'),
        ('Keep fps', 'keep_fps'),
        ('Keep audio', 'keep_audio'),
        ('Keep frames', 'keep_frames'),
        ('NSFW filter', 'nsfw_filter')
    ]

    for i, (text, attr) in enumerate(switches_right):
        value = ctk.BooleanVar(value=getattr(modules.globals, attr, False))
        switch = ctk.CTkSwitch(root, text=text, variable=value, cursor='hand2',
                               command=lambda a=attr, v=value: setattr(modules.globals, a, v.get()))
        switch.place(relx=0.55, rely=y_start + i*y_increment, relwidth=0.4)

    # Face Enhancer switch (modified)
    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui.get('face_enhancer', False))
    enhancer_switch = ctk.CTkSwitch(root, text='Face Enhancer', variable=enhancer_value, cursor='hand2',
                                    command=lambda: update_tumbler('face_enhancer', enhancer_value.get()))
    enhancer_switch.place(relx=0.55, rely=y_start + 5*y_increment, relwidth=0.4)

    # Outline frame for mouth mask and dropdown
    outline_frame = ctk.CTkFrame(root, fg_color="transparent", border_width=2, border_color="grey")
    outline_frame.place(relx=0.02, rely=y_start + 5.8*y_increment, relwidth=0.96, relheight=0.06)

    # Mouth mask switch
    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    mouth_mask_switch = ctk.CTkSwitch(outline_frame, text='Mouth mask | Feather, Height, Width ->', variable=mouth_mask_var, cursor='hand2',
                                    command=lambda: setattr(modules.globals, 'mouth_mask', mouth_mask_var.get()))
    mouth_mask_switch.place(relx=0.03, rely=0.1, relwidth=0.8, relheight=0.8)

    # Feather ratio dropdown
    mask_feather_ratio_var = ctk.StringVar(value="8")
    mask_feather_ratio_size_dropdown = ctk.CTkOptionMenu(outline_frame, values=["1","2","3","4","5","6","7","8","9","10"],
                                            variable=mask_feather_ratio_var,
                                            command=mask_feather_ratio_size)
    mask_feather_ratio_size_dropdown.place(relx=0.65, rely=0.1, relwidth=0.1, relheight=0.8)

    # Down size dropdown
    mask_down_size_var = ctk.StringVar(value="5")
    mask_down_size_dropdown = ctk.CTkOptionMenu(outline_frame, values=["-15","-14","-13","-12","-11","-10","-9","-8","-7","-6","-5","-4","-3","-2","-1","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"],
                                            variable=mask_down_size_var,
                                            command=mask_down_size)
    mask_down_size_dropdown.place(relx=0.76, rely=0.1, relwidth=0.1, relheight=0.8)

    # Size dropdown
    mask_size_var = ctk.StringVar(value="11")
    mask_size_dropdown = ctk.CTkOptionMenu(outline_frame, values=["5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"],
                                            variable=mask_size_var,
                                            command=mask_size)
    mask_size_dropdown.place(relx=0.87, rely=0.1, relwidth=0.1, relheight=0.8)



    # Bottom buttons
    button_width = 0.18  # Width of each button
    button_height = 0.05  # Height of each button
    button_y = 0.85  # Y position of the buttons
    space_between = (1 - (button_width * 5)) / 6  # Space between buttons

    start_button = ctk.CTkButton(root, text='Start', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=space_between, rely=button_y, relwidth=button_width, relheight=button_height)

    stop_button = ctk.CTkButton(root, text='Destroy', cursor='hand2', command=lambda: destroy())
    stop_button.place(relx=space_between*2 + button_width, rely=button_y, relwidth=button_width, relheight=button_height)

    preview_button = ctk.CTkButton(root, text='Preview', cursor='hand2', command=lambda: toggle_preview())
    preview_button.place(relx=space_between*3 + button_width*2, rely=button_y, relwidth=button_width, relheight=button_height)

    live_button = ctk.CTkButton(root, text='Live', cursor='hand2', command=lambda: webcam_preview(), fg_color="green", hover_color="dark green")
    live_button.place(relx=space_between*4 + button_width*3, rely=button_y, relwidth=button_width, relheight=button_height)

    preview_size_var = ctk.StringVar(value="640x360")
    preview_size_dropdown = ctk.CTkOptionMenu(root, values=["640x360","854x480", "960x540", "1280x720", "1920x1080"],
                                              variable=preview_size_var,
                                              command=update_preview_size,
                                              fg_color="green", button_color="dark green", button_hover_color="forest green")
    preview_size_dropdown.place(relx=space_between*5 + button_width*4, rely=button_y, relwidth=button_width, relheight=button_height)

    # Status and donate labels
    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=0.05, rely=0.93, relwidth=0.9)

    donate_label = ctk.CTkLabel(root, text='iRoopDeepFaceCam', justify='center', cursor='hand2')
    donate_label.place(relx=0.05, rely=0.96, relwidth=0.9)
    donate_label.configure(text_color=ctk.ThemeManager.theme.get('URL').get('text_color'))
    donate_label.bind('<Button>', lambda event: webbrowser.open('https://buymeacoffee.com/ivideogameboss'))

    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='select an source image', initialdir=RECENT_DIRECTORY_SOURCE, filetypes=[img_ft])
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=RECENT_DIRECTORY_TARGET, filetypes=[img_ft, vid_ft])
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save image output file', filetypes=[img_ft], defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save video output file', filetypes=[vid_ft], defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    ''' Check if the target is NSFW.
    TODO: Consider to make blur the target.
    '''
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame
    if type(target) is str: # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray: # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy: destroy(to_quit=False) # Do not need to destroy the window frame if the target is NSFW
        update_status('Processing ignored!')
        return True
    else: return False


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
      return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width  / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return cv2.resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status('Processing...')
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        
        source_image_left = None  # Left source face image
        source_image_right = None  # Right source face image
        
        # Initialize variables for the selected face/s image. 
        # Source image can have one face or two faces we simply detect face from left of frame
        # then right of frame. This insures we always have a face to work with
        if source_image_left is None and modules.globals.source_path:
            source_image_left = get_one_face_left(cv2.imread(modules.globals.source_path))
        if source_image_right is None and modules.globals.source_path:
            source_image_right = get_one_face_right(cv2.imread(modules.globals.source_path))

        # no face found
        if source_image_left is None:
            print('No face found in source image')
            return
        
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            temp_frame = frame_processor.process_frame([source_image_left,source_image_right],
                temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status('Processing succeed!')
        PREVIEW.deiconify()

def webcam_preview():
    if modules.globals.source_path is None:
        return

    global preview_label, PREVIEW, ROOT, camera

    # Set initial size of the preview window
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 360

    camera = cv2.VideoCapture(0)
    update_camera_resolution()

    # Configure the preview window
    PREVIEW.deiconify()
    PREVIEW.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
    preview_label.configure(width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

    source_image_left = None
    source_image_right = None
    
    if modules.globals.source_path:
        source_image_left = get_one_face_left(cv2.imread(modules.globals.source_path))
        source_image_right = get_one_face_right(cv2.imread(modules.globals.source_path))
    
    if source_image_left is None:
        print('No face found in source image')
        return

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        temp_frame = frame.copy()

        if modules.globals.live_mirror:
            temp_frame = cv2.flip(temp_frame, 1)

        for frame_processor in frame_processors:
            temp_frame = frame_processor.process_frame([source_image_left, source_image_right], temp_frame)

        # Get current preview window size
        current_width = PREVIEW.winfo_width()
        current_height = PREVIEW.winfo_height()

        # Resize the processed frame to fit the current preview window size
        temp_frame = fit_image_to_preview(temp_frame, current_width, current_height)

        image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ctk.CTkImage(image, size=(current_width, current_height))
        preview_label.configure(image=image, width=current_width, height=current_height)
        ROOT.update()

        if PREVIEW.state() == 'withdrawn':
            break

    camera.release()
    PREVIEW.withdraw()


def fit_image_to_preview(image, preview_width, preview_height):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if preview_width / preview_height > aspect_ratio:
        new_height = preview_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = preview_width
        new_height = int(new_width / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Create a black canvas of the size of the preview window
    canvas = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)

    # Calculate position to paste the resized image
    y_offset = (preview_height - new_height) // 2
    x_offset = (preview_width - new_width) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return canvas



def update_preview_size(*args):
    global PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, camera
    size = preview_size_var.get().split('x')
    PREVIEW_DEFAULT_WIDTH = int(size[0])
    PREVIEW_DEFAULT_HEIGHT = int(size[1])
    
    if camera is not None and camera.isOpened():
        update_camera_resolution()
    
    if PREVIEW.state() == 'normal':
        update_preview()

def update_camera_resolution():
    global camera, PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT
    if camera is not None and camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_DEFAULT_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_DEFAULT_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, 60)  # You may want to make FPS configurable as well
        
def mask_size(*args):
    size = mask_size_var.get()
    modules.globals.mask_size = int(size)

def mask_down_size(*args):
    size = mask_down_size_var.get()
    modules.globals.mask_down_size = int(size)

def mask_feather_ratio_size(*args):
    size = mask_feather_ratio_var.get()
    modules.globals.mask_feather_ratio = int(size)


    
