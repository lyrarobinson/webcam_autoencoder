import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

def crop_to_square(frame):
    height, width = frame.shape[:2]
    size = min(height, width)
    top_left_x = (width - size) // 2
    top_left_y = (height - size) // 2
    cropped_frame = frame[top_left_y:top_left_y + size, top_left_x:top_left_x + size]
    
    # Resize the cropped square to 224x224
    resized_frame = cv2.resize(cropped_frame, (224, 224), interpolation=cv2.INTER_AREA)
    return resized_frame

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = crop_to_square(frame)
    # Now frame is a 224x224 grayscale image
    input_grid = cv2.resize(frame, (7, 7), interpolation=cv2.INTER_AREA).astype(np.float32)
    input_grid /= 255.0
    return input_grid.flatten().reshape(1, 49)

def adjust_contrast_brightness(image, alpha=5.0, beta=100):
    """ Adjust contrast (alpha) and brightness (beta) more aggressively """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def main():
    custom_objects = {'mse': MeanSquaredError()}
    model = load_model('autoencoder.h5', custom_objects=custom_objects)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    pygame.init()
    window_size = (800, 800)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Autoencoder Live Feed")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        input_grid = preprocess_frame(frame)
        output_image = model.predict(input_grid)[0]
        output_image = (output_image * 255).astype(np.uint8)

        print(f"Debug: Output shape before any processing: {output_image.shape}")
        print(f"Debug: Sample pixel values (before RGB conversion): {output_image[0:5, 0:5]}")

        if output_image.ndim == 3 and output_image.shape[2] == 1:
            output_image = np.repeat(output_image, 3, axis=2)
            print(f"Debug: Converted to RGB with shape: {output_image.shape}")
            print(f"Debug: Sample pixel values (after RGB conversion): {output_image[0:5, 0:5, :]}")

        if output_image.ndim != 3 or output_image.shape[2] != 3:
            raise ValueError("Output image must be 3D with 3 channels for RGB.")

        # Aggressively adjust brightness and contrast
        output_image = adjust_contrast_brightness(output_image, alpha=5.0, beta=100)

        pygame_image = pygame.surfarray.make_surface(output_image)
        pygame_image = pygame.transform.scale(pygame_image, window_size)
        screen.blit(pygame_image, (0, 0))
        pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
