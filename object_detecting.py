import cv2


def get_model_input_size(model):
    input_size = model.get_image_size()
    print("Input Size: ", input_size[0])  # get the input size of the model


def image_pred(model, img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # cv2.imshow("image", img)
    # cv2.waitKey(100)
    # cv2.waitKey(0)
    pred, file_path = model.predict(img, "./data/images/")

    if len(pred) == 0:
        print("No specfied object is detected!")

    return pred, file_path
