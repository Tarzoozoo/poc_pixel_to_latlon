from image2geo import img2geo

IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 1024
FOV_WIDTH    = 22.9
FOV_HEIGHT   = 18.4
ALTITUDE     = 1000
TILT_ANGLE   = 0
HEADING      = 0
CAM_LAT      = 13.756300
CAM_LNG      = 100.501800

imggeo = img2geo(
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    FOV_WIDTH,
    FOV_HEIGHT,
    ALTITUDE,
    TILT_ANGLE,
    HEADING,
    CAM_LAT,
    CAM_LNG
)
bbox = [700, 572, 580, 452]  # [Xmax, Ymax, Xmin, Ymin]
TL_bbox = [515, 422, 365, 302]
TR_bbox = [915, 422, 765, 302]
BL_bbox = [515, 722, 365, 602]
BR_bbox = [915, 722, 765, 602]
info = imggeo.get_bbox_info(TL_bbox)
print(f'Input lat: {13.756300} Input lng: {100.501800}')
print(f'Output lat: {info.lat} Output lng: {info.lng}')
# print(info.lat, info.lng, info.width, info.height, info.size, info.distance)
