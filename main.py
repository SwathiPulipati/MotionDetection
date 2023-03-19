import cv2
import numpy as np
import pyautogui
cam = cv2.VideoCapture(0)
static_back = None
motion = 0
minimum = 50
mog = cv2.createBackgroundSubtractorMOG2()


while cam.isOpened():
    input = cam.read()[1]
    frame = cv2.flip(input, 1)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

# eroding background subtracted video
    bgs = mog.apply(blur)
    erosion_size = 2
    erosion_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))

    eroded = cv2.erode(bgs, element)

# thresholding for white
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grayscale, 249, 255, cv2.THRESH_BINARY)[1]

# bitwise_and for motion and light
    combined = cv2.bitwise_and(eroded, thresh)
# showing dot for lights
    contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < minimum:
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        print((x,y))
        print((x/600)*1920, (y/400)*1080)
        pyautogui.moveTo((x/600)*1920, (y/400)*1080, .05)
        # pyautogui.dragTo((x / 600) * 1920, (y / 400) * 1080)

    eroded = np.stack((eroded, )*3, -1)
    bgs = np.stack((bgs,) * 3, -1)
    frame_top = np.hstack((frame,eroded))

    thresh = np.stack((thresh,) * 3, -1)
    combined = np.stack((combined,) * 3, -1)
    frame_bottom = np.hstack((thresh,combined))
    composite_frame = np.vstack((frame_top,frame_bottom))

    cv2.imshow('composite', composite_frame)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
