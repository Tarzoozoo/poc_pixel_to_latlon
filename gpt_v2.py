from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# from location import detected_color_locations, pixel_to_latlon
from test_new_gimbal_2 import (
    pixel_to_latlon_k40t_thermal,
)

# Load image
image_path = "red1.jpg"  # Replace with your image path
# image_path = "ad.jpeg"
# image_path = "alt50.png"
# image_path = "alt80.png"
image = Image.open(image_path).convert("RGB")

# show the image
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")
plt.show()

image_np = np.array(image)


def find_center_of_detected_pixels(detected_pixels):
    """
    Find the center (centroid) of all detected colored pixels

    Args:
        detected_pixels: numpy array of shape (n, 2) containing [row, col] coordinates

    Returns:
        tuple: (center_row, center_col) as floats
    """
    if len(detected_pixels) == 0:
        return None, None

    center_row = np.mean(detected_pixels[:, 0])  # Average of all row coordinates
    center_col = np.mean(detected_pixels[:, 1])  # Average of all column coordinates

    return center_row, center_col


def find_zone_centers_connected_components(color_mask, min_area=50):
    """
    หา center ของแต่ละโซนโดยใช้ connected components

    Args:
        color_mask: boolean mask ของพิกเซลที่เป็นสี
        min_area: ขนาดพื้นที่ขั้นต่ำของโซนที่จะนับ (pixels)

    Returns:
        list: [(zone_id, center_row, center_col, area), ...]
    """
    # หา connected components
    num_labels, labels_im = cv2.connectedComponents(color_mask.astype(np.uint8))

    zone_centers = []

    for label in range(1, num_labels):  # Skip background (label 0)
        # หาพิกเซลของโซนนี้
        mask = labels_im == label
        area = np.sum(mask)

        # กรองโซนที่เล็กเกินไป
        if area < min_area:
            continue

        # หา center ของโซน
        rows, cols = np.where(mask)
        center_row = np.mean(rows)
        center_col = np.mean(cols)

        zone_centers.append((label, center_row, center_col, area))

    return zone_centers, labels_im


def visualize_zone_centers(
    image_np, zone_centers, labels_im=None, method="connected_components"
):
    """
    แสดงผลโซนและ center ของแต่ละโซน
    """
    output_image = image_np.copy()

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 128, 128),
        (255, 128, 0),
    ]

    for i, (zone_id, center_row, center_col, area_or_pixels) in enumerate(zone_centers):
        color = colors[i % len(colors)]

        # วาดจุด center
        cv2.circle(output_image, (int(center_col), int(center_row)), 8, color, -1)
        cv2.circle(
            output_image, (int(center_col), int(center_row)), 12, (255, 255, 255), 2
        )

        # เขียนข้อมูลโซน
        text = f"Zone {zone_id}"
        cv2.putText(
            output_image,
            text,
            (int(center_col) + 15, int(center_row) - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        coord_text = f"({center_row:.1f}, {center_col:.1f})"
        cv2.putText(
            output_image,
            coord_text,
            (int(center_col) + 15, int(center_row) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        area_text = f"Area: {area_or_pixels}"
        cv2.putText(
            output_image,
            area_text,
            (int(center_col) + 15, int(center_row) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return output_image


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
hfov_deg = (40.0,)  # FOV กว้าง 40 องศา
vfov_deg = (32.0,)  # FOV สูง 30 องศา
gimbal_yaw_deg = (0,)  # หันหน้าเหนือ
gimbal_pitch_deg = (-89.9,)  # ชี้ตรงลง (active stabilization)
gimbal_roll_deg = (0,)  # level
ground_elevation_m = (0,)  # ระดับน้ำทะเล
utm_zone = "32647"  # UTM zone สำหรับประเทศไทย

print(len(detected_pixels), "detected pixels")
zone_centers_cc, labels_im = find_zone_centers_connected_components(
    color_mask, min_area=100
)
print("=== Connected Components Method ===")
for zone_id, center_row, center_col, area in zone_centers_cc:
    print(
        f"Zone {zone_id}: Center at ({center_row:.2f}, {center_col:.2f}), Area: {area} pixels"
    )

    # Find locations of detected colors
    lat, lon = pixel_to_latlon_k40t_thermal(
        drone_lat=13.7563,
        drone_lon=100.5018,
        drone_altitude_m=90,
        hfov_deg=40.0,
        vfov_deg=32.0,
        gimbal_yaw_deg=0,  # Adjust based on gimbal orientation
        gimbal_pitch_deg=-89.9,  # Nearly straight down
        gimbal_roll_deg=0,
        image_width=640,
        image_height=512,
        pixel_x=center_col,
        pixel_y=center_row,
        ground_elevation_m=0,  # Ground elevation
        utm_zone="32647",  # UTM zone for Thailand
    )
    print(f"Zone {zone_id} GPS: Latitude {lat:.6f}, Longitude {lon:.6f}")

# สร้างรูปภาพแสดงผล
output_with_centers = visualize_zone_centers(image_np, zone_centers_cc, labels_im)

# สร้างรูปที่ 4 แสดง Zone centers พร้อม lat/lon
output_with_gps = image_np.copy()

# วาด zone centers พร้อมข้อมูล GPS
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
          (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)]

for i, (zone_id, center_row, center_col, area) in enumerate(zone_centers_cc):
    color = colors[i % len(colors)]
    
    # แปลงเป็นพิกัด lat/lon
    lat, lon = pixel_to_latlon_k40t_thermal(
        drone_lat=13.7563,
        drone_lon=100.5018,
        drone_altitude_m=90,
        hfov_deg=40.0,
        vfov_deg=32.0,
        gimbal_yaw_deg=0,  # Adjust based on gimbal orientation
        gimbal_pitch_deg=-89.9,  # Nearly straight down
        gimbal_roll_deg=0,
        image_width=640,
        image_height=512,
        pixel_x=center_col,
        pixel_y=center_row,
        ground_elevation_m=0,  # Ground elevation
        utm_zone="32647",  # UTM zone for Thailand
    )
    
    # วาดจุด center
    cv2.circle(output_with_gps, (int(center_col), int(center_row)), 10, color, -1)
    cv2.circle(output_with_gps, (int(center_col), int(center_row)), 15, (255, 255, 255), 3)
    
    # เขียนข้อมูล Zone
    zone_text = f"Zone {zone_id}"
    cv2.putText(output_with_gps, zone_text, (int(center_col) + 20, int(center_row) - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    # เขียน Latitude
    lat_text = f"Lat: {lat:.6f}"
    cv2.putText(output_with_gps, lat_text, (int(center_col) + 20, int(center_row)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # เขียน Longitude
    lon_text = f"Lon: {lon:.6f}"
    cv2.putText(output_with_gps, lon_text, (int(center_col) + 20, int(center_row) + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # เขียน Area
    area_text = f"Area: {area}px"
    cv2.putText(output_with_gps, area_text, (int(center_col) + 20, int(center_row) + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

# สร้างรูปภาพแสดงผล
output_with_centers = visualize_zone_centers(image_np, zone_centers_cc, labels_im)

# แสดงผล 4 รูป
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('1. Original Image', fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(color_mask, cmap='gray')
plt.title('2. Color Mask', fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(output_with_centers)
plt.title('3. Zone Centers (Pixel Coordinates)', fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(output_with_gps)
plt.title('4. Zone Centers with GPS Coordinates', fontsize=14, fontweight='bold')
plt.axis('off')



plt.tight_layout()
plt.show()

# สร้างสรุปข้อมูลทั้งหมด
print(f"\n=== Summary ===")
print(f"Total zones detected: {len(zone_centers_cc)}")
print(f"Total colored pixels: {np.sum(color_mask)}")
print(f"Image dimensions: {image_np.shape[1]} x {image_np.shape[0]}")

# สร้างรูปภาพสำหรับแสดงข้อมูลสรุป (ใช้แทน output_image เดิมของคุณ)
summary_image = image_np.copy()

# เพิ่มข้อมูลโซนลงในรูปภาพสรุป
for i, (zone_id, center_row, center_col, area) in enumerate(zone_centers_cc):
    cv2.putText(summary_image, f"Zone {zone_id} Center: ({center_row:.1f}, {center_col:.1f})", 
                (10, 120 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
# # Print detected locations
# for i, (lat, lon) in enumerate(locations):
#     print(f"Color {i + 1}: Latitude {lat:.6f}, Longitude {lon:.6f}")

# # Create output image with labels
# output_image = image_np.copy()
# output_image[color_mask] = [255, 0, 0]  # Mark colored pixels in red

# # Label connected regions
# num_labels, labels_im = cv2.connectedComponents(color_mask.astype(np.uint8))

# # Calculate total pixels in the image
# total_pixels = image_np.shape[0] * image_np.shape[1]

# # Calculate total percentage of colored area
# total_color_pixels = np.sum(color_mask)
# total_color_percentage = (total_color_pixels / total_pixels) * 100

# # Display total percentage on the image
# cv2.putText(
#     output_image,
#     f"Total Color Area: {total_color_percentage:.2f}%",
#     (10, 30),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     0.7,
#     (255, 255, 255),
#     2,
#     cv2.LINE_AA,
# )

# # Display detected colored pixels per total pixels
# cv2.putText(
#     output_image,
#     f"Detected Pixels: {total_color_pixels}/{total_pixels}",
#     (10, 60),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     0.7,
#     (255, 255, 255),
#     2,
#     cv2.LINE_AA,
# )

# # Draw labels with percentage
# for label in range(1, num_labels):  # Skip background label 0
#     mask = labels_im == label
#     y, x = np.where(mask)
#     cx, cy = int(np.mean(x)), int(np.mean(y))
#     region_pixels = np.sum(mask)
#     percentage = (region_pixels / total_pixels) * 100
#     cv2.putText(
#         output_image,
#         f"Color {label} ({percentage:.2f}%)",
#         (cx, cy),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 255, 0),
#         1,
#         cv2.LINE_AA,
#     )

# Group detected points by connected regions and estimate center location
# region_centers = []
# for label in range(1, num_labels):  # Skip background label 0
#     mask = labels_im == label
#     y, x = np.where(mask)
#     cx, cy = int(np.mean(x)), int(np.mean(y))  # Center pixel of the region

#     # Estimate center location of the region
#     lat, lon = pixel_to_latlon(
#         center_lat,
#         center_lon,
#         altitude_m,
#         fov_width_deg,
#         fov_height_deg,
#         image_np.shape[1],
#         image_np.shape[0],
#         cx,
#         cy,
#     )
#     region_centers.append((label, lat, lon))

# # Print region center locations
# for label, lat, lon in region_centers:
#     print(f"Region {label}: Center Latitude {lat:.6f}, Longitude {lon:.6f}")

# # Label detected points directly at their center
# for label, lat, lon in region_centers:
#     cx, cy = int(np.mean(np.where(labels_im == label)[1])), int(
#         np.mean(np.where(labels_im == label)[0])
#     )
#     cv2.putText(
#         output_image,
#         f"Region {label}",
#         (cx, cy),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 255, 0),
#         1,
#         cv2.LINE_AA,
#     )

# # Label detected points with region number and lat/lon directly at their center
# for label, lat, lon in region_centers:
#     cx, cy = int(np.mean(np.where(labels_im == label)[1])), int(
#         np.mean(np.where(labels_im == label)[0])
#     )
#     cv2.putText(
#         output_image,
#         f"Region {label}",
#         (cx, cy - 10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 255, 0),
#         1,
#         cv2.LINE_AA,
#     )
#     cv2.putText(
#         output_image,
#         f"({lat:.6f}, {lon:.6f})",
#         (cx, cy + 10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (255, 255, 255),
#         1,
#         cv2.LINE_AA,
#     )

# # Display region center locations on the image
# for label, lat, lon in region_centers:
#     cv2.putText(
#         output_image,
#         f"Region {label}: ({lat:.6f}, {lon:.6f})",
#         (10, 90 + label * 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (255, 255, 255),
#         1,
#         cv2.LINE_AA,
#     )

# # Display result
# plt.figure(figsize=(10, 6))
# plt.imshow(output_image)
# plt.axis("off")
# plt.title("Detected Non-Grayscale Colors")
# plt.show()
