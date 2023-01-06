from numpy import pi, sin, cos, arctan2, sqrt

radius = 6370E3

def measure(x1, y1, x2, y2):
    lat1, lng1 = y1*pi/180., x1*pi/180.
    lat2, lng2 = y2*pi/180., x2*pi/180.
    sin_lat1, cos_lat1 = sin(lat1), cos(lat1)
    sin_lat2, cos_lat2 = sin(lat2), cos(lat2)
    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = cos(delta_lng), sin(delta_lng)
    d = arctan2(sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                (cos_lat1 * sin_lat2 -
                sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)
    return radius * d


_rad = pi/180.

def raddist(lon1, lat1, lon2, lat2):
    '''distance on unit-radius sphere, arguments in radians'''
    delta_lng = lon2 - lon1
    sin_lat1 = sin(lat1)
    cos_lat1 = cos(lat1)
    sin_lat2 = sin(lat2)
    cos_lat2 = cos(lat2)
    cos_delta_lng = cos(delta_lng)
    sin_delta_lng = sin(delta_lng)
    cc = cos_lat2 * cos_delta_lng
    y1 = cos_lat2 * sin_delta_lng
    y = cos_lat1 * sin_lat2 
    y -= sin_lat1 * cc
    y *= y
    y += y1*y1
    y = sqrt(y)
    x = sin_lat1 * sin_lat2
    x += cos_lat1 * cc
    d = arctan2(y, x)
    return d
