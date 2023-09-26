from motion_detection import picam2
from object_detecting import image_pred
from object_detected import save_to_csv
from edgetpumodel import EdgeTPUModel
import time

# Define variables for use use in handlers
Image_Buffer = None  # declare image buffer for captured image
Pred = None  # the prediction in each frame
File_Path = "./data/images"  # the file path of the captured image if saved 
Idle_Max = 5  # where 5 frame no motion detected, then object detection again
TIME_DELAY = 1 # setting time sleep (in sec) in idle stage

# A simple Finite State Machine Class
class fsm:
    """
    A simple state machine implementation.

    This class allows you to manage a finite state machine with various states 
    and associated handler functions.

    Attributes:
        handlers (dict): A dictionary mapping state names to handler functions.
    """
    def __init__(self):
        self.handlers = {}

    def add_state(self, name, handler):
        self.handlers[name] = handler

    def run(self, startingState, idle_count=0, motion_flag=False, object_flag=False):
        handler = self.handlers[startingState]
        while True:
            (newState, idle_count, motion_flag, object_flag) = handler(idle_count, 
                                                                          motion_flag, 
                                                                          object_flag)
            handler = self.handlers[newState]

# Define state handlers and variables

def state_motion_detecting(idle_count, motion_flag, object_flag):
    print("Now is motion detecting stage.")
    motion_flag = picam.motion_detection()  # check motion is detected or not
    if motion_flag:
        Image_Buffer = (
            picam.capture_image()
        )  # return a fresh capture in ndarray image
        return(("object_detecting", idle_count, motion_flag, object_flag))
    else:
        return(("motion_not_detected", idle_count, motion_flag, object_flag))

def state_motion_not_detected(idle_count, motion_flag, object_flag):
    print("Now is motion not detected stage.")
    idle_count += 1
    if (idle_count <= Idle_Max):
        return(("state_idle", idle_count, motion_flag, object_flag))
    else:
        return(("object_detecting", idle_count, motion_flag, object_flag))

def state_object_detecting(idle_count, motion_flag, object_flag):
    print("Now is object detecting stage.")
    global Image_Buffer  # explicitly declare Image_Buffer is global variable
    global File_Path  # explicitly declare File_Path is global variable
    print(idle_count)
    Image_Buffer = picam.capture_image() if (idle_count - 1) == Idle_Max else Image_Buffer
    (Pred, File_Path) = image_pred(
        yolov5s_224_edge, Image_Buffer
    )  # output of detected bbox and file name
    
    if (idle_count -1 ) == Idle_Max:
        idle_count = 0  # reset counter only if it reach max

    object_flag = True if len(Pred) != 0 else False
    motion_flag = False  # reset flag

    if object_flag:
        return(("object_detected", idle_count, motion_flag, object_flag))
    else:
        return(("motion_detecting", idle_count, motion_flag, object_flag))

def state_object_detected(idle_count, motion_flag, object_flag):
    print("Now is object detected stage.")
    global File_Path
    save_to_csv(File_Path)  # save the summary of detection results in csv
    return(("state_idle", idle_count, motion_flag, object_flag))

def state_idle(idle_count, motion_flag, object_flag):
    print("Now is idle stage.")
    time.sleep(TIME_DELAY)  # cooling time after detected object if needed
    if object_flag:
        object_flag = False  # reset flag
        return(("motion_detecting", idle_count, motion_flag, object_flag))
    elif idle_count == 0:
        print("Initiatizing...")
        return(("motion_detecting", idle_count, motion_flag, object_flag))  
    else:
        print("Idle stage {} of {}.".format(idle_count, Idle_Max))
        return(("motion_detecting", idle_count, motion_flag, object_flag))

# Initiation of Object Detection State Machine
def fsm_init(obj):

    obj.add_state("motion_detecting", state_motion_detecting )
    obj.add_state("motion_not_detected", state_motion_not_detected )
    obj.add_state("object_detecting", state_object_detecting )
    obj.add_state("object_detected", state_object_detected )
    obj.add_state("state_idle", state_idle )

# Initization of edge TPU and loading model
"""
Inputs:
    - model_file: path to edgetpu-compiled tflite file
    - names_file: yaml names file (yolov5 format)
    - conf_thresh: detection threshold
    - iou_thresh: NMS threshold
    - filter_classes: only output certain classes
    - agnostic_nms: use class-agnostic NMS
    - max_det: max number of detections
"""
yolov5s_224_edge = EdgeTPUModel(
    model_file="yolov5s-int8-224_edgetpu.tflite",  # tflite model file
    names_file="data/coco.yaml",  # classes
    conf_thresh=0.25,
    iou_thresh=0.45,
    filter_classes=None,  # show specific classes
    agnostic_nms=False,
    max_det=1000,
)  # maximum detection objects

if __name__ == "__main__":  # if run the below if directly exceute this py
    # Initization of PiCamera
    picam = picam2()
    # Initization of object detection state machine
    PiObjectDetection = fsm()
    fsm_init(PiObjectDetection)
    # Start runing the object detection
    PiObjectDetection.run("state_idle")