import time
import cv2
import numpy as np
from skimage.util import img_as_float, img_as_ubyte


def show_in_moved_window(win_name, img, x, y):
    """ Show an image in a window at a given position. """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)


def capture_from_camera_and_show_images(alpha=0.95, T=0.1, A=0.05):
    print("Starting image capture")

    # Open camera
    url = 0  # set to DroidCam URL if needed
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")

    # Get first frame -> background
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        exit()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    I_background = img_as_float(frame_gray)

    # FPS tracking
    start_time = time.time()
    n_frames = 0
    stop = False

    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Convert to grayscale float
        I_new = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        I_new = img_as_float(I_new)

        # Absolute difference
        dif_img = np.abs(I_new - I_background)

        # Thresholding -> binary image
        binary_img = dif_img > T
        binary_img_uint8 = img_as_ubyte(binary_img)

        # Foreground ratio
        F = np.sum(binary_img) / binary_img.size

        # FPS
        n_frames += 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Draw info on frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(new_frame, f"fps: {fps}", (20, 40), font, 1, (0, 255, 0), 2)
        cv2.putText(new_frame, f"Changed pixels: {np.sum(binary_img)}", (20, 80), font, 0.7, (0, 255, 255), 2)
        cv2.putText(new_frame, f"Change %: {F:.3f}", (20, 120), font, 0.7, (255, 255, 0), 2)
        cv2.putText(new_frame, f"Diff min: {dif_img.min():.3f}", (20, 160), font, 0.7, (255, 0, 0), 2)
        cv2.putText(new_frame, f"Diff max: {dif_img.max():.3f}", (20, 200), font, 0.7, (0, 0, 255), 2)
        cv2.putText(new_frame, f"Diff mean: {dif_img.mean():.3f}", (20, 240), font, 0.7, (0, 255, 0), 2)

        # Alarm
        if F > A:
            cv2.putText(new_frame, "!!! CHANGE DETECTED !!!", (100, 300), font, 1.2, (0, 0, 255), 3)

        # Display
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Background', I_background, 600, 10)
        show_in_moved_window('Difference', dif_img, 1200, 10)
        show_in_moved_window('Binary', binary_img_uint8, 1800, 10)

        # Update background
        I_background = alpha * I_background + (1 - alpha) * I_new

        # Stop with 'q'
        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Try with alpha=0.95, T=0.1, A=0.05
    capture_from_camera_and_show_images(alpha=0.95, T=0.35, A=0.05)
