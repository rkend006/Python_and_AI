import numpy as np
from training import get_model, load_trained_model, compile_model
import cv2

# Load the trained model
model = get_model()
load_trained_model(model)

# Get frontal face haar cascade
face_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_default.xml')

santa_filter = cv2.imread('filters/santa_filter.png', -1)
hat = cv2.imread('filters/hat2.png', -1)
glasses = cv2.imread('filters/glasses.png', -1)

# Get webcam
camera = cv2.VideoCapture(0)

while True:
    grab_trueorfalse, img = camera.read()

    # Convert RGB data from webcam to Grayscale and detect face with cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # get grayscaled face
        img_copy = np.copy(img)
        img_copy_1 = np.copy(img)
        roi_color = img_copy_1[y:y+h, x:x+w]

        # Width of region where face is detected
        width_original = roi_gray.shape[1]
        # Height of region where face is detected
        height_original = roi_gray.shape[0]
        # Resize image to size 96x96
        img_gray = cv2.resize(roi_gray, (96, 96))
        img_gray = img_gray/255         # Normalize the image data

        # Model takes input of shape = [batch_size, height, width, no. of channels]
        img_model = np.reshape(img_gray, (1, 96, 96, 1))
        # Predict keypoints for the current input
        keypoints = model.predict(img_model)[0]

        # Keypoints are saved as (x1, y1, x2, y2, ......)
        # Read alternate elements starting from index 0
        x_coords = keypoints[0::2]
        # Read alternate elements starting from index 1
        y_coords = keypoints[1::2]

        # Denormalize x-coordinate
        x_coords_denormalized = (x_coords+0.5)*width_original
        # Denormalize y-coordinate
        y_coords_denormalized = (y_coords+0.5)*height_original

        # Particular keypoints for scaling and positioning of the filter
        left_lip_coords = (int(x_coords_denormalized[11]), int(
            y_coords_denormalized[11]))

        right_lip_coords = (int(x_coords_denormalized[12]), int(
            y_coords_denormalized[12]))

        top_lip_coords = (int(x_coords_denormalized[13]), int(
            y_coords_denormalized[13]))

        bottom_lip_coords = (
            int(x_coords_denormalized[14]), int(y_coords_denormalized[14]))

        left_eye_coords = (int(x_coords_denormalized[3]), int(
            y_coords_denormalized[3]))

        right_eye_coords = (
            int(x_coords_denormalized[5]), int(y_coords_denormalized[5]))

        brow_coords = (int(x_coords_denormalized[6]), int(
            y_coords_denormalized[6]))

        # Scale filter according to keypoint coordinates
        beard_width = right_lip_coords[0] - left_lip_coords[0]
        glasses_width = right_eye_coords[0] - left_eye_coords[0]

        # Used for transparency overlay of filter using the alpha channel
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)

        # Beard filter
        santa_filter_resized = cv2.resize(santa_filter, (beard_width*3, 150))
        sw, sh, sc = santa_filter_resized.shape

        for i in range(0, sw):       # Overlay the filter based on the alpha channel
            for j in range(0, sh):
                if santa_filter_resized[i, j][3] != 0:
                    img_copy[top_lip_coords[1]+i+y-20,
                             left_lip_coords[0]+j+x-60] = santa_filter_resized[i, j]

        # Hat filter
        hat_resized = cv2.resize(hat, (w, w))
        hw, hh, hc = hat_resized.shape

        for i in range(0, hw):       # Overlay the filter based on the alpha channel
            for j in range(0, hh):
                if hat_resized[i, j][3] != 0:
                    img_copy[i+y-brow_coords[1]*2, j+x -
                             left_eye_coords[0]*1 + 20] = hat_resized[i, j]

        # Glasses filter
        glasses_resized = cv2.resize(glasses, (glasses_width*2, 150))
        gw, gh, gc = glasses_resized.shape

        for i in range(0, gw):       # Overlay the filter based on the alpha channel
            for j in range(0, gh):
                if glasses_resized[i, j][3] != 0:
                    img_copy[brow_coords[1]+i+y-50,
                             left_eye_coords[0]+j+x-60] = glasses_resized[i, j]

        # Revert back to BGR
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGRA2BGR)

        # Output with the filter placed on the face
        cv2.imshow('Output', img_copy)

    # If 'e' is pressed, stop reading and break the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
