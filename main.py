from transitions import Machine
from motion_detection import picam2
from object_detecting import image_pred
from object_detected import save_to_csv
from edgetpumodel import EdgeTPUModel
import time


class PiObjectDetection:
    """
    A class for object detection.

    Attributes:
        states (List[str]): The various states in object detection

    """

    def __init__(self):
        self.states = [
            "motion_detect",
            "motion_not_detected",
            "object_detecting",
            "object_detected",
            "idle",
        ]
        self.machine = Machine(model=self, states=self.states, initial="idle")

        # Initialize context variable
        self.context = {}
        self.context["count"] = 0

        # Initialize the transition flow from stage to stage
        self.machine.add_transition(
            "ch_to_motion_n_detected",
            "motion_detect",
            "motion_not_detected",
            before="store_data",
        )
        self.machine.add_transition(
            "ch_to_object_detecting",
            ["motion_detect", "motion_not_detected"],
            "object_detecting",
            before="store_data",
        )
        self.machine.add_transition(
            "ch_to_object_detected",
            "object_detecting",
            "object_detected",
            before="store_data",
        )
        self.machine.add_transition(
            "ch_to_idle",
            ["object_detected", "motion_not_detected"],
            "idle",
            before="store_data",
        )
        self.machine.add_transition(
            "ch_to_motion_detect", "idle", "motion_detect", before="store_data"
        )
        self.machine.add_transition(
            "ch_to_motion_detect",
            "object_detecting",
            "motion_detect",
            before="store_data",
        )

    def store_data(self):
        self.context["previous_state"] = self.state

    # Actions on entering into different states
    def on_enter_idle(self):
        print(
            "Previous state is {}, now is {} now.".format(
                self.context["previous_state"], self.state
            )
        )

    def on_enter_motion_detect(self):
        print(
            "Previous state is {}, now is {} state now.".format(
                self.context["previous_state"], self.state
            )
        )

    def on_enter_motion_n_detected(self):
        print(
            "Previous state is {}, now is {} state now.".format(
                self.context["previous_state"], self.state
            )
        )

    def on_enter_object_detecting(self):
        print(
            "Previous state is {}, now is {} state now.".format(
                self.context["previous_state"], self.state
            )
        )

    def on_enter_object_detected(self):
        print(
            "Previous state is {}, now is {} state now.".format(
                self.context["previous_state"], self.state
            )
        )

    # Loop for object detection
    def run(self):
        while True:
            PiRun.ch_to_motion_detect()  # start the intial stage of the loop
            if picam.motion_detection():  # check motion is detected or not
                np_image = (
                    picam.capture_image()
                )  # return a fresh capture in ndarray image
                PiRun.context["count"] = 0
                PiRun.ch_to_object_detecting()
                pred, file_path = image_pred(
                    yolov5s_224_edge, np_image
                )  # output of detected bbox and file name
                if len(pred) == 0:  # nothing detected
                    pass
                    # time.sleep(10)  # cooling time if needed
                else:  # object detected
                    PiRun.ch_to_object_detected()
                    save_to_csv(
                        file_path
                    )  # save the summary of detection results in csv
                    PiRun.ch_to_idle()
                    time.sleep(1)  # cooling time if needed
            else:
                PiRun.ch_to_motion_n_detected()
                PiRun.context["count"] += 1
                if (
                    PiRun.context["count"] <= 5
                ):  # having no motion detected for 5 rounds, enter object detecting state
                    PiRun.ch_to_idle()
                    print("Idle Count: {} of 5".format(PiRun.context["count"]))
                    time.sleep(1)

                else:
                    PiRun.context["count"] = 0
                    np_image = (
                        picam.capture_image()
                    )  # return a fresh capture in ndarray image
                    PiRun.ch_to_object_detecting()
                    pred, file_path = image_pred(
                        yolov5s_224_edge, np_image
                    )  # output of detected bbox and file name
                    if len(pred) == 0:  # nothing detected
                        pass
                        # time.sleep(10)  # cooling time if needed
                    else:  # object detected
                        PiRun.ch_to_object_detected()
                        save_to_csv(
                            file_path
                        )  # save the summary of detection results in csv
                        PiRun.ch_to_idle()
                        time.sleep(1)  # cooling time if needed


# Initization of PiCamera
picam = picam2()

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

# File path for saving the annotation file
file_path = "./data/images"

if __name__ == "__main__":  # if run the below if directly exceute this py
    # Kick of the machine from motion detect state
    PiRun = PiObjectDetection()
    # Start run the object detection
    PiRun.run()
