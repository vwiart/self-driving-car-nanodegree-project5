import cv2
import numpy as np

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def as_tuple(self):
        return (self.x, self.y)


class Box(object):

    def __init__(self, top_left, bottom_right, border=6):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.border = border

    def draw(self, img, color=(0, 0, 255)):
        img = np.copy(img)
        cv2.rectangle(img,
                      self.top_left.as_tuple(),
                      self.bottom_right.as_tuple(),
                      color,
                      self.border)
        return img

def test_draw_box():
    img = cv2.imread('test_images/test1.jpg', cv2.IMREAD_COLOR)

    box = Box(Point(800,400), Point(950, 500))
    img = box.draw(img)
    cv2.imwrite('output_images/draw_box.jpg', img)