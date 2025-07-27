import cv2
import numpy as np

# Get file from arguments or define it directly in agruments[2]
import sys
if len(sys.argv) < 2:
    print("Usage: python run.py <image_path>")
    sys.exit(1)
image_path = sys.argv[1]

print(f"Processing image: {image_path}")
# Load the image
image = cv2.imread(image_path)  # Replace with your image path
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    sys.exit(1)

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("HSV values of the image:")
print(hsv)  # Debugging: Print HSV values to verify ranges

# Debugging: Show the HSV image to inspect values
cv2.imshow('HSV Image', hsv)
print("Press any key to close the HSV Image window.")
cv2.waitKey(5000)  # Wait for 5 seconds or until a key is pressed
cv2.destroyWindow('HSV Image')  # Close only the HSV Image window

# Define the range for red color in HSV
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Adjust the red color ranges if necessary
print("Adjust the red color ranges if the mask is blank or incorrect.")
print("Current ranges:")
print(f"Lower Red 1: {lower_red1}, Upper Red 1: {upper_red1}")
print(f"Lower Red 2: {lower_red2}, Upper Red 2: {upper_red2}")

# Optional: Save the HSV image for inspection
cv2.imwrite('hsv_image.jpg', hsv)

def detect_red_color(image):
    """Detect red color in the given image."""
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to refine the mask
    kernel = np.ones((2, 2), np.uint8)  # Use a smaller kernel
    red_mask = cv2.erode(red_mask, kernel, iterations=2)  # Erode more to remove noise
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)  # Dilate more to restore shape

    # Detect contours for smallest red areas
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Detected red area at: x={x}, y={y}, width={w}, height={h}")  # Log location

        # Draw contours on the original image
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)  # Green color for contours

    # Bitwise-AND mask and original image to highlight red areas
    red_highlighted = cv2.bitwise_and(image, image, mask=red_mask)

    return red_mask, red_highlighted

def detect_brown_color(image):
    """Detect brown color in the given image."""
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for brown color in HSV
    lower_brown = np.array([118, 85, 79])  # Adjusted for brown
    upper_brown = np.array([180, 255, 255])  # Adjusted for brown

    # Create mask for brown color
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Apply morphological operations to refine the mask
    kernel = np.ones((2, 2), np.uint8)  # Use a smaller kernel
    brown_mask = cv2.erode(brown_mask, kernel, iterations=2)  # Erode more to remove noise
    brown_mask = cv2.dilate(brown_mask, kernel, iterations=2)  # Dilate more to restore shape

    # Detect contours for brown areas
    contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Detected brown area at: x={x}, y={y}, width={w}, height={h}")  # Log location

        # Draw contours on the original image
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)  # Green color for contours

    # Bitwise-AND mask and original image to highlight brown areas
    brown_highlighted = cv2.bitwise_and(image, image, mask=brown_mask)

    return brown_mask, brown_highlighted

def detect_non_grayscale(image):
    """Detect areas in the image that are not grayscale using histograms."""
    # Split the image into its B, G, R channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Calculate histograms for each channel
    b_hist = cv2.calcHist([b_channel], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
    r_hist = cv2.calcHist([r_channel], [0], None, [256], [0, 256])

    # Compare histograms to detect non-grayscale areas
    diff_mask = cv2.absdiff(cv2.absdiff(r_channel, g_channel), b_channel)
    _, non_grayscale_mask = cv2.threshold(diff_mask, 1, 255, cv2.THRESH_BINARY)

    # Detect contours for non-grayscale areas
    contours, _ = cv2.findContours(non_grayscale_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Detected non-grayscale area at: x={x}, y={y}, width={w}, height={h}")  # Log location

        # Draw contours on the original image
        cv2.drawContours(image, [contour], -1, (255, 0, 0), 1)  # Blue color for contours

    # Bitwise-AND mask and original image to highlight non-grayscale areas
    non_grayscale_highlighted = cv2.bitwise_and(image, image, mask=non_grayscale_mask)

    return non_grayscale_mask, non_grayscale_highlighted

def detect_specific_color(image, target_color):
    """Detect a specific color in the given image."""
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert the target color from HEX to HSV
    target_bgr = np.array([int(target_color[i:i+2], 16) for i in (4, 2, 0)], dtype=np.uint8)
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # Define a range around the target color in HSV
    lower_bound = np.array([max(0, target_hsv[0] - 10), max(0, target_hsv[1] - 40), max(0, target_hsv[2] - 40)])
    upper_bound = np.array([min(180, target_hsv[0] + 10), min(255, target_hsv[1] + 40), min(255, target_hsv[2] + 40)])

    # Create mask for the specific color
    specific_color_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Detect contours for the specific color areas
    contours, _ = cv2.findContours(specific_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Detected specific color area at: x={x}, y={y}, width={w}, height={h}")  # Log location

        # Draw contours on the original image
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)  # Green color for contours

    # Bitwise-AND mask and original image to highlight specific color areas
    specific_color_highlighted = cv2.bitwise_and(image, image, mask=specific_color_mask)

    return specific_color_mask, specific_color_highlighted

# Call the function to detect red color
red_mask, red_highlighted = detect_red_color(image)

# Optional: Show the result
cv2.imshow('Red Mask', red_mask)  # Debugging: Show the mask
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Red Color Detection', red_highlighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('red_detected.jpg', red_highlighted)

# Call the function to detect brown color
brown_mask, brown_highlighted = detect_brown_color(image)

# Optional: Show the result
cv2.imshow('Brown Mask', brown_mask)  # Debugging: Show the mask
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Brown Color Detection', brown_highlighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('brown_detected.jpg', brown_highlighted)

# Call the function to detect non-grayscale areas
non_grayscale_mask, non_grayscale_highlighted = detect_non_grayscale(image)

# Optional: Show the result
cv2.imshow('Non-Grayscale Mask', non_grayscale_mask)  # Debugging: Show the mask
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Non-Grayscale Detection', non_grayscale_highlighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('non_grayscale_detected.jpg', non_grayscale_highlighted)

# Call the function to detect the specific color #76554f
specific_color_mask, specific_color_highlighted = detect_specific_color(image, "#76554f")

# Optional: Show the result
cv2.imshow('Specific Color Mask', specific_color_mask)  # Debugging: Show the mask
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Specific Color Detection', specific_color_highlighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('specific_color_detected.jpg', specific_color_highlighted)
