from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from location import detected_color_locations, pixel_to_latlon

# Load image
image_path = "red1.jpg"  # Replace with your image path
# image_path = "ad.jpeg"
# image_path = "alt50.png"
# image_path = "alt80.png"
image = Image.open(image_path).convert("RGB")

# show the image
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.show()

image_np = np.array(image)


# convert any not pure red to grayscale
def convert_non_red_to_grayscale(image_np):
    # Create a mask for non-red pixels
    red_mask = (image_np[:, :, 0] > 150) & (image_np[:, :, 1] < 100) & (image_np[:, :, 2] < 100)
    
    # Create a grayscale version of the image
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    
    # Apply the mask to keep red pixels and convert others to grayscale
    output_image = np.where(red_mask[..., None], image_np, gray_3ch)
    
    return output_image
# image_np = convert_non_red_to_grayscale(image_np)

# Convert to grayscale
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Compute difference from grayscale to find colored pixels
color_diff = cv2.absdiff(image_np, gray_3ch)
color_mask = np.any(color_diff > 10, axis=-1)  # Threshold to detect color

# Extract pixel coordinates of detected colors
detected_pixels = np.column_stack(np.where(color_mask))

# Example parameters for location calculation
center_lat = 13.7563  # Replace with actual latitude
center_lon = 100.5018  # Replace with actual longitude
altitude_m = 100  # Replace with actual altitude
fov_width_deg = 28.1  # Replace with actual FOV width
fov_height_deg = 21.3  # Replace with actual FOV height

print (len(detected_pixels), "detected pixels")
# Find locations of detected colors
locations = detected_color_locations(
    center_lat, center_lon, altitude_m, fov_width_deg, fov_height_deg,
    image_np.shape[1], image_np.shape[0], detected_pixels
)

# Print detected locations
# for i, (lat, lon) in enumerate(locations):
#     print(f"Color {i + 1}: Latitude {lat:.6f}, Longitude {lon:.6f}")

# Create output image with labels
output_image = image_np.copy()
output_image[color_mask] = [255, 0, 0]  # Mark colored pixels in red

# Label connected regions
num_labels, labels_im = cv2.connectedComponents(color_mask.astype(np.uint8))

# Calculate total pixels in the image
total_pixels = image_np.shape[0] * image_np.shape[1]

# Calculate total percentage of colored area
total_color_pixels = np.sum(color_mask)
total_color_percentage = (total_color_pixels / total_pixels) * 100

# Display total percentage on the image
cv2.putText(output_image, f"Total Color Area: {total_color_percentage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (255, 255, 255), 2, cv2.LINE_AA)

# Display detected colored pixels per total pixels
cv2.putText(output_image, f"Detected Pixels: {total_color_pixels}/{total_pixels}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (255, 255, 255), 2, cv2.LINE_AA)

# Draw labels with percentage
for label in range(1, num_labels):  # Skip background label 0
    mask = labels_im == label
    y, x = np.where(mask)
    cx, cy = int(np.mean(x)), int(np.mean(y))
    region_pixels = np.sum(mask)
    percentage = (region_pixels / total_pixels) * 100
    cv2.putText(output_image, f"Color {label} ({percentage:.2f}%)", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Group detected points by connected regions and estimate center location
region_centers = []
for label in range(1, num_labels):  # Skip background label 0
    mask = labels_im == label
    y, x = np.where(mask)
    cx, cy = int(np.mean(x)), int(np.mean(y))  # Center pixel of the region

    # Estimate center location of the region
    lat, lon = pixel_to_latlon(
        center_lat, center_lon, altitude_m, fov_width_deg, fov_height_deg,
        image_np.shape[1], image_np.shape[0], cx, cy
    )
    region_centers.append((label, lat, lon))

# Print region center locations
for label, lat, lon in region_centers:
    print(f"Region {label}: Center Latitude {lat:.6f}, Longitude {lon:.6f}")

# Label detected points directly at their center
for label, lat, lon in region_centers:
    cx, cy = int(np.mean(np.where(labels_im == label)[1])), int(np.mean(np.where(labels_im == label)[0]))
    cv2.putText(output_image, f"Region {label}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Label detected points with region number and lat/lon directly at their center
for label, lat, lon in region_centers:
    cx, cy = int(np.mean(np.where(labels_im == label)[1])), int(np.mean(np.where(labels_im == label)[0]))
    cv2.putText(output_image, f"Region {label}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(output_image, f"({lat:.6f}, {lon:.6f})", (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Display region center locations on the image
for label, lat, lon in region_centers:
    cv2.putText(output_image, f"Region {label}: ({lat:.6f}, {lon:.6f})", (10, 90 + label * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Display result
plt.figure(figsize=(10, 6))
plt.imshow(output_image)
plt.axis('off')
plt.title('Detected Non-Grayscale Colors')
plt.show()