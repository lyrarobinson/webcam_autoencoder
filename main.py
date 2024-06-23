import pygame
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

def main():
    # Load the model with an explicitly defined loss function
    autoencoder = load_model('autoencoder.h5', custom_objects={'mse': MeanSquaredError()})

    # Pygame setup
    pygame.init()
    width, height = 700, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('webcam_autoencoder')
    background_color = (0, 0, 0)

    cap = cv2.VideoCapture(0)  # 0 is typically the default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    def normalize_image(image):
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        # Scale to [0, 255]
        image = (image * 255).astype(np.uint8)
        return image

    def update_display(data):
        screen.fill(background_color)
        normalized_data = data.flatten().reshape(1, 49) / 100.0
        predicted_image = autoencoder.predict(normalized_data)
        predicted_image = predicted_image.squeeze()
        predicted_image = normalize_image(predicted_image)

        if len(predicted_image.shape) == 2:
            predicted_image = np.stack((predicted_image,) * 3, axis=-1)

        predicted_image = np.transpose(predicted_image, (1, 0, 2))
        pygame_image = pygame.surfarray.make_surface(predicted_image)
        pygame_image = pygame.transform.scale(pygame_image, (width, height))
        screen.blit(pygame_image, (0, 0))
        pygame.display.flip()

    def process_frame(frame):
        # Assuming the default resolution is higher than 980x980
        height, width = frame.shape[:2]
        # Calculate margins to crop the center square
        top = (height - 980) // 2
        left = (width - 980) // 2
        cropped_frame = frame[top:top+980, left:left+980]
        resized_frame = cv2.resize(cropped_frame, (7, 7))  # Resize to 7x7 for the model
        return resized_frame

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        processed_frame = process_frame(frame)  # Process to square and resize
        update_display(np.array(processed_frame, dtype=float))  # Pass the processed frame for display

        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                running = False

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
