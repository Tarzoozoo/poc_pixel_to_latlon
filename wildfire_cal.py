import math
from typing import Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class bboxInfo:
    Xmax: float
    Ymax: float
    Xmin: float
    Ymin: float

@dataclass
class pointInfo:
    bbox: bboxInfo = None
    lat: float = 0.0
    lng: float = 0.0
    width: float = 0.0
    height: float = 0.0
    size: float = 0.0
    distance: float = 0.0

class img2geo:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        fov_width: float,
        fov_height: float,
        altitude: float,
        tilt_angle: float,  # Total Pitch
        heading: float,  # Total Yaw
        camera_lat: float,
        camera_lng: float,
        station_lat: float = 17.083783116300882, 
        station_lng: float = 102.70642427905587,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.fov_width = fov_width
        self.fov_height = fov_height
        self.altitude = altitude
        self.tilt_angle = tilt_angle
        self.heading = heading
        self.camera_lat = camera_lat
        self.camera_lng = camera_lng
        self.station_lat = station_lat
        self.station_lng = station_lng
        self.pixel_center_x = image_width / 2
        self.pixel_center_y = image_height / 2

    def __center(self, bbox: bboxInfo) -> Tuple[float, float]:
        center_x = (bbox.Xmax + bbox.Xmin) / 2  # pixel X
        center_y = (bbox.Ymax + bbox.Ymin) / 2  # pixel Y
        return center_x, center_y

    def __pixel_to_lat_lng_with_tilt(
        self, pixel_x: int, pixel_y: int
    ) -> Tuple[float, float]:

        fov_width_rad = math.radians(self.fov_width)
        fov_height_rad = math.radians(self.fov_height)
        tilt_angle_rad = math.radians(self.tilt_angle)
        heading_rad = math.radians(self.heading)

        # Calculate the ground dimensions of the image (meter)
        ground_width = 2 * self.altitude * math.tan(fov_width_rad / 2)
        ground_height = 2 * self.altitude * math.tan(fov_height_rad / 2)
        
        # Calculate the pixel size on the ground (meter)
        pixel_size_x = ground_width / self.image_width
        pixel_size_y = ground_height / self.image_height

        # Calculate the offset of the pixel from the image center
        offset_x = (pixel_x - self.image_width / 2) * pixel_size_x
        offset_y = -(pixel_y - self.image_height / 2) * pixel_size_y # Convert image coordinate to LAT/LON coordinate

        # TODO: Adjust for tilt angle
        offset_y += self.altitude * math.tan(tilt_angle_rad)

        # TODO: Rotate the offsets based on the heading angle
        R = np.array([
            [math.cos(heading_rad), math.sin(heading_rad)],
            [-math.sin(heading_rad), math.cos(heading_rad)]
        ])
        offset_vec = np.array([offset_x, offset_y])
        rotate_vec = R @ offset_vec
        rotated_x, rotated_y = rotate_vec

        # Convert ground offsets to latitude and longitude (degrees)
        earth_radius = 6378137  # Earth's radius in meters
        delta_lat = (rotated_y / earth_radius) * (180 / math.pi)
        delta_lng = (
            rotated_x / (earth_radius * math.cos(math.radians(self.camera_lat)))
        ) * (180 / math.pi)

        pixel_lat = self.camera_lat + delta_lat
        pixel_lng = self.camera_lng + delta_lng

        return pixel_lat, pixel_lng


    def get_bbox_info(self, bbox) -> pointInfo:
        point_info = pointInfo()
        point_info.bbox = bboxInfo(
            Xmax=bbox[0], Ymax=bbox[1], Xmin=bbox[2], Ymin=bbox[3]
        )

        center_x, center_y = self.__center(point_info.bbox)
        point_info.lat, point_info.lng = self.__pixel_to_lat_lng_with_tilt(
            center_x, center_y
        )
        return point_info
