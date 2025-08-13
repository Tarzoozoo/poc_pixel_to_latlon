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
        tilt_angle: float,  # Pitch camera
        heading: float,  # Yaw drone + camera
        camera_lat: float,
        camera_lng: float,
        station_lat: float = 17.083783116300882,
        station_lng: float = 102.70642427905587,
    ):
        """
        Initialize the img2geo object with image dimensions, field of view, altitude, tilt angle, heading, and camera coordinates.

        Args:
            image_width (int): Width of the image in pixels.
            image_height (int): Height of the image in pixels.
            fov_width (float): Field of View (FOV) in the horizontal direction (degrees).
            fov_height (float): Field of View (FOV) in the vertical direction (degrees).
            altitude (float): Altitude of the camera (meters).
            tilt_angle (float): Tilt angle of the camera (degrees).
            heading (float): Heading angle of the camera (degrees).
            camera_lat (float): Latitude of the camera.
            camera_lng (float): Longitude of the camera.
        """
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
        self.center_lat, self.center_lng = self.__find_center_lat_lng()

    def __find_center_lat_lng(self) -> Tuple[float, float]:
        """
        คำนวณพิกัดละติจูดและลองจิจูดของจุดศูนย์กลางจากพิกัดละติจูด, ลองจิจูด, ความสูง, การเอียง และการหันหน้า
        """
        # รัศมีโลก (เมตร)
        R = 6378137

        # แปลงการเอียงและการหันหน้าเป็น radians
        tilt_rad = math.radians(self.tilt_angle)
        heading_rad = math.radians(self.heading)

        # คำนวณระยะทางในแนวราบที่ต้องการ
        distance = self.altitude * math.tan(tilt_rad)

        # คำนวณพิกัดใหม่
        delta_lat = (distance / R) * (180 / math.pi)
        delta_lng = (distance / (R * math.cos(math.radians(self.camera_lat)))) * (
            180 / math.pi
        )

        center_lat = self.camera_lat + delta_lat * math.cos(heading_rad)
        center_lng = self.camera_lng + delta_lng * math.sin(heading_rad)

        return center_lat, center_lng

    # from input bbox to following information
    # 1.) center location (lat, lng) of bbox
    # 2.) size of bbox in meters
    # 3.) distance in meter from specific point (lat, lng) to center of bbox
    def __center(self, bbox: bboxInfo) -> Tuple[float, float]:
        """
        Calculate the center latitude and longitude of a bounding box.

        Args:
            bbox (bboxInfo): Bounding box coordinates.

        Returns:
            tuple: A tuple containing the center latitude and longitude.
        """
        # center_lat = (bbox.Ymax + bbox.Ymin) / 2  # ❌ ผิด - นี่คือ pixel coordinate
        # center_lng = (bbox.Xmax + bbox.Xmin) / 2  # ❌ ผิด - นี่คือ pixel coordinate
        # return center_lat, center_lng
        center_x = (bbox.Xmax + bbox.Xmin) / 2  # pixel X
        center_y = (bbox.Ymax + bbox.Ymin) / 2  # pixel Y
        return center_x, center_y

    def __pixel_to_lat_lng_with_tilt(
        self, pixel_x: int, pixel_y: int
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to latitude and longitude considering tilt and heading angles.
        Args:
            pixel_x (int): X coordinate of the pixel.
            pixel_y (int): Y coordinate of the pixel.
            image_width (int): Width of the image in pixels.
            image_height (int): Height of the image in pixels.
            fov_width (float): Field of View (FOV) in the horizontal direction (degrees).
            fov_height (float): Field of View (FOV) in the vertical direction (degrees).
            altitude (float): Altitude of the camera (meters).
            tilt_angle (float): Tilt angle of the camera (degrees).
            heading (float): Heading angle of the camera (degrees).
            camera_lat (float): Latitude of the camera.
            camera_lng (float): Longitude of the camera.
        Returns:
            tuple: Latitude and longitude of the pixel.
        """
        # Convert angles to radians
        fov_width_rad = math.radians(self.fov_width)
        fov_height_rad = math.radians(self.fov_height)
        tilt_angle_rad = math.radians(self.tilt_angle)
        heading_rad = math.radians(self.heading)

        # Calculate the ground dimensions of the image
        ground_width = 2 * self.altitude * math.tan(fov_width_rad / 2)
        ground_height = 2 * self.altitude * math.tan(fov_height_rad / 2)

        # Calculate the pixel size on the ground
        pixel_size_x = ground_width / self.image_width
        pixel_size_y = ground_height / self.image_height

        print ('pixel_x: ', pixel_x)
        print ('self.image_width / 2: ', self.image_width / 2)
        print ('pixel_size_x: ', pixel_size_x)
        # Calculate the offset of the pixel from the image center
        offset_x = (pixel_x - self.image_width / 2) * pixel_size_x
        # offset_y = (pixel_y - self.image_height / 2) * pixel_size_y
        offset_y = -(pixel_y - self.image_height / 2) * pixel_size_y

        
        # Adjust for tilt angle
        offset_y += self.altitude * math.tan(tilt_angle_rad)
        # offset_y -= self.altitude * math.tan(tilt_angle_rad)
        print ('offset_x: ', offset_x)
        print ('offset_y: ', offset_y)
        print ('heading :', heading_rad)
        # Rotate the offsets based on the heading angle
        rotated_x = offset_x * math.cos(heading_rad) - offset_y * math.sin(heading_rad)
        rotated_y = offset_x * math.sin(heading_rad) + offset_y * math.cos(heading_rad)

        # Convert ground offsets to latitude and longitude
        earth_radius = 6378137  # Earth's radius in meters
        delta_lat = (rotated_y / earth_radius) * (180 / math.pi)
        delta_lng = (
            rotated_x / (earth_radius * math.cos(math.radians(self.camera_lat)))
        ) * (180 / math.pi)

        print ('delta_lat', delta_lat)
        print ('delta_lng', delta_lng)
        pixel_lat = self.camera_lat + delta_lat
        pixel_lng = self.camera_lng + delta_lng

        return pixel_lat, pixel_lng

    def __size_with_tilt(
        self, pixel_width: int, pixel_height: int
    ) -> Tuple[float, float]:
        # def __size_with_tilt(pixel_width, pixel_height, image_width, image_height, fov_width, fov_height, altitude, tilt_angle):
        """
        Calculate the real-world size (in meters) from pixel dimensions, considering the tilt angle.

        Parameters:
            pixel_width (int): Width of the object in pixels.
            pixel_height (int): Height of the object in pixels.
            image_width (int): Width of the image in pixels.
            image_height (int): Height of the image in pixels.
            fov_width (float): Field of View (FOV) in the horizontal direction (degrees).
            fov_height (float): Field of View (FOV) in the vertical direction (degrees).
            altitude (float): Altitude of the camera (meters).
            tilt_angle (float): Tilt angle of the camera (degrees).

        Returns:
            tuple: Real-world width and height of the object (in meters).
        """

        # Convert FOV and tilt angle to radians
        fov_width_rad = math.radians(self.fov_width)
        fov_height_rad = math.radians(self.fov_height)
        tilt_angle_rad = math.radians(self.tilt_angle)

        # Adjust altitude for tilt angle
        effective_altitude = self.altitude * math.cos(tilt_angle_rad)

        # Calculate ground dimensions of the image
        ground_width = (
            2 * self.altitude * math.tan(fov_width_rad / 2)
        )  # Horizontal ground width remains the same
        ground_height = (
            2 * effective_altitude * math.tan(fov_height_rad / 2)
        )  # Adjusted for tilt

        # Calculate GSD (Ground Sampling Distance)
        gsd_x = ground_width / self.image_width  # Meters per pixel in the X direction
        gsd_y = ground_height / self.image_height  # Meters per pixel in the Y direction

        # Calculate real-world size
        width = pixel_width * gsd_x
        height = pixel_height * gsd_y

        return width, height

    def __distance(self, lat, lng):
        # def calculate_distance(lat1, lon1, lat2=17.083783116300882, lon2=102.70642427905587):
        """
        Calculate the distance between two geographical points using the Haversine formula.

        Args:
            lat (float): Latitude of the point.
            lon (float): Longitude of the point.

        Returns:
            float: Distance in meters between the two points.
        """
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000  # Radius of the Earth in meters
        phi1 = radians(self.station_lat)
        phi2 = radians(lat)
        delta_phi = radians(lat - self.station_lat)
        delta_lambda = radians(lng - self.station_lng)

        a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def get_bbox_info(self, bbox) -> pointInfo:
        """
        Get the center location, size, and distance from a specific point to the center of the bounding box.

        Args:
            bbox (list): Bounding box coordinates in the format [Xmax, Ymax, Xmin, Ymin].

        Returns:
            tuple: A tuple containing the center latitude, center longitude, width, height, area, and distance.
        """
        point_info = pointInfo()
        point_info.bbox = bboxInfo(
            Xmax=bbox[0], Ymax=bbox[1], Xmin=bbox[2], Ymin=bbox[3]
        )

        center_x, center_y = self.__center(point_info.bbox)
        print ('Center_x: ', center_x)
        print ('Center_y: ', center_y)
        point_info.lat, point_info.lng = self.__pixel_to_lat_lng_with_tilt(
            center_x, center_y
        )
        # point_info.width, point_info.height = self.__size_with_tilt(
        #     point_info.bbox.Xmax - point_info.bbox.Xmin,
        #     point_info.bbox.Ymax - point_info.bbox.Ymin,
        # )
        # point_info.size = point_info.width * point_info.height  # Area in square meters
        # point_info.distance = self.__distance(point_info.lat, point_info.lng)
        return point_info
