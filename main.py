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
		# Normalize the 7x7 data for model input
		normalized_data = data / 255.0  # Assuming data is in the range [0, 255]
		normalized_data = normalized_data.flatten().reshape(1, 49)  # Model expects shape (1, 49)
		
		# Predict the output using the autoencoder
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

		# Divide the 980x980 frame into a 7x7 grid of squares each 140x140
		grid_size = 140
		squares = []
		for y in range(0, 980, grid_size):
			for x in range(0, 980, grid_size):
				square = cropped_frame[y:y+grid_size, x:x+grid_size]
				average_intensity = np.mean(square)
				squares.append(average_intensity)
		
		# Reshape the list of averages into a 7x7 grid
		processed_frame = np.array(squares).reshape((7, 7))
		return processed_frame

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
