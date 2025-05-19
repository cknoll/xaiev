import cv2
import numpy as np
# import csv
import os
import random
import hashlib
import math

from sympy import true
# The parent class for all shapes, use this class to change the path to save pictures
# and the number of images to be generated
class Geometry():
    def __init__(self):
        # define the size of the picture,
        # the folder for storing images and the number of images to be generated
        self.pic_size = 224  # 200x200 px image
        self.output_folder = ""
        self.background_folder = ""
        self.mask_folder = ""
        self.num = 50
        self.color = []
        self.name = ''

    # common method for generating background noise
    def draw_background(self, i):
        background_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        background_image_name = self.name + f"_{i+1:03d}.png"
        image_path = os.path.join(self.background_folder, background_image_name)
        cv2.imwrite(image_path, background_img)
        return background_img

    def draw_geometry(self, i):
        return None, None

    def save_images(self):

        tags = []  # List containing all the angle and center points of the rectangles
        number_of_duplicates_found = 0
        i = 0
        while i < self.num:
            drawn_image, mask, tag = self.draw_geometry(i)
            # Check boundaries
            duplicate_found_flag = False
            if not self.check_boundaries(drawn_image, self.color[i]):
                # Check duplicity
                individual_tag = tag
                for item in tags:
                    # Check the tag for duplicity
                    if individual_tag == item:
                        duplicate_found_flag = True
                        number_of_duplicates_found += 1

                if not duplicate_found_flag:
                    tags.append(individual_tag)
                    # Save valid image
                    i += 1
                    image_name = self.name + f"_{i:03d}.png"
                    image_path = os.path.join(self.output_folder, image_name)
                    mask_name = self.name + f"_{i:03d}.png"
                    mask_path = os.path.join(self.mask_folder, mask_name)
                    cv2.imwrite(image_path, drawn_image)
                    cv2.imwrite(mask_path, mask)
        print(f"In the dataset generation there were discarded {number_of_duplicates_found} duplicate images")

    def check_boundaries(self, image, color):
        '''
            We define a border where it should always be "clean". Otherwise,
            we assume that we have our rectangle outside of the boundaries
            Return:
                False -> Figure inside boundaries
                True -> Figure outside boundaries
        '''
        # Check if the rectangle is out of boundaries
        shape_rgb = np.array(color)
        edges = np.vstack([image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]])
        return np.any(np.all(edges == shape_rgb, axis=1))  # check boundary pixels for match of color



# class for generating rectangles
class Rectangle(Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/01_rectangle"
            self.background_folder = "imgs_background/test/01_rectangle"
            self.mask_folder = "imgs_mask/test/01_rectangle"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/01_rectangle"
            self.background_folder = "imgs_background/train/01_rectangle"
            self.mask_folder = "imgs_mask/train/01_rectangle"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        self.rectangle_width = []
        self.rectangle_height = []
        # randomly generate height/width of all rectangles and make sure height!=width(range:30-80)
        while len(self.rectangle_width) < self.num:
            w = random.randint(30, 80)
            h = random.randint(30, 80)
            if abs(w - h) > 20:
                self.rectangle_width.append(w)
                self.rectangle_height.append(h)
        # randomly generate colors for all rectangles
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'rectangle'

    # Draw a rectangle
    def draw_rectangle(self, base_image, rectangle_width, rectangle_height, color, center_width, center_height, angle):
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        rotation_matrix = cv2.getRotationMatrix2D((center_width, center_height), angle, 1)

        # define the four vertexes of the rectangle
        rect_points = np.array([
            [center_width - rectangle_width // 2, center_height - rectangle_height // 2],
            [center_width + rectangle_width // 2, center_height - rectangle_height // 2],
            [center_width + rectangle_width // 2, center_height + rectangle_height // 2],
            [center_width - rectangle_width // 2, center_height + rectangle_height // 2]
        ])

        # rotate the vertexes
        rotated_points = cv2.transform(np.array([rect_points]), rotation_matrix)[0]

        # Use the vertexes to draw a rectangle and fill with a random color
        rotated_points = rotated_points.astype(int)
        return cv2.fillPoly(base_image, [rotated_points], color), cv2.fillPoly(mask_background, [rotated_points], color = (255, 255, 255) ) 

    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        # generate a random angle and a random center point position for the rectangle and draw
        angle = random.randint(0, 180)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        tag = [angle, center_width, center_height]
        drawn_image, mask = self.draw_rectangle(base_image, self.rectangle_width[i], self.rectangle_height[i],
                                          tuple(self.color[i].tolist()), center_width, center_height, angle)
        return drawn_image, mask, tag
    # function for calling draw function and save images


# class for generating rectangles(basically same as rectangle, except that height = width)
class Square(Geometry):

    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/02_square"
            self.background_folder = "imgs_background/test/02_square"
            self.mask_folder = "imgs_mask/test/02_square"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/02_square"
            self.background_folder = "imgs_background/train/02_square"
            self.mask_folder = "imgs_mask/train/02_square"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        self.size = np.random.randint(30, 80, size=self.num)
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'square'

    def draw_square(self, base_image, size, color, center_width, center_height, angle):
        # Draw a rectangle
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        rotation_matrix = cv2.getRotationMatrix2D((center_width, center_height), angle, 1)

        rect_points = np.array([
            [center_width - size // 2, center_height - size // 2],
            [center_width + size // 2, center_height - size // 2],
            [center_width + size // 2, center_height + size // 2],
            [center_width - size // 2, center_height + size // 2]
        ])

        rotated_points = cv2.transform(np.array([rect_points]), rotation_matrix)[0]

        rotated_points = rotated_points.astype(int)
        return cv2.fillPoly(base_image, [rotated_points], color), cv2.fillPoly(mask_background, [rotated_points], color = (255, 255, 255) ) 

    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        angle = random.randint(0, 180)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        tag = [angle, center_width, center_height]
        drawn_image, mask = self.draw_square(base_image, self.size[i], tuple(self.color[i].tolist()),
                                       center_width, center_height, angle)
        return drawn_image, mask, tag

# class for generating rectangles
class Circle(Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/03_circle"
            self.background_folder = "imgs_background/test/03_circle"
            self.mask_folder = "imgs_mask/test/03_circle"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/03_circle"
            self.background_folder = "imgs_background/train/03_circle"
            self.mask_folder = "imgs_mask/train/03_circle"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        # randomly generate the radii and colors for all images
        self.radius = np.random.randint(30, 80, size=self.num)
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'circle'

    # draw the circle(since it makes no difference rotating the circle, we only need different center points)
    def draw_circle(self, base_image, radius, color, center_width, center_height):
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        return cv2.circle(base_image, (center_width, center_height), radius, color, -1), cv2.circle(mask_background, (center_width, center_height), radius, (255,255,255), -1)

    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        tag = [self.radius[i], center_width, center_height]
        drawn_image, mask = self.draw_circle(base_image, self.radius[i], tuple(self.color[i].tolist()),
                                       center_width, center_height)
        return drawn_image, mask, tag

    # it does the same thing as the previous save_images function

class Triangle(Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/04_triangle"
            self.background_folder = "imgs_background/test/04_triangle"
            self.mask_folder = "imgs_mask/test/04_triangle"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/04_triangle"
            self.background_folder = "imgs_background/train/04_triangle"
            self.mask_folder = "imgs_mask/train/04_triangle"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        # randomly generate the radii and colors for all images
        self.bottom_len = np.random.randint(60, 130, size=self.num)
        self.height = np.random.randint(60, 130, size=self.num)
        self.shift = np.random.randint(-70, 70, size=self.num)
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'triangle'

    def draw_triangle(self, base_image, bottom_len, shift, height, color, center_width, center_height, angle):
        # Draw a rectangle
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        rotation_matrix = cv2.getRotationMatrix2D((center_width, center_height), angle, 1)

        rect_points = np.array([
            [center_width, center_height],
            [center_width + bottom_len, center_height],
            [center_width + shift, center_height + height]
        ])

        rotated_points = cv2.transform(np.array([rect_points]), rotation_matrix)[0]

        rotated_points = rotated_points.astype(int)
        return cv2.fillPoly(base_image, [rotated_points], color), cv2.fillPoly(mask_background, [rotated_points], color = (255, 255, 255) ) 

    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        angle = random.randint(0, 180)
        tag = [self.bottom_len[i], self.height[i], self.shift[i], center_width, center_height, angle]
        drawn_image, mask = self.draw_triangle(base_image, self.bottom_len[i], self.shift[i], self.height[i],
                                         tuple(self.color[i].tolist()), center_width, center_height, angle)
        return drawn_image, mask, tag

class Trapezoid(Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/05_trapezoid"
            self.background_folder = "imgs_background/test/05_trapezoid"
            self.mask_folder = "imgs_mask/test/05_trapezoid"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/05_trapezoid"
            self.background_folder = "imgs_background/train/05_trapezoid"
            self.mask_folder = "imgs_mask/train/05_trapezoid"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        # randomly generate the radii and colors for all images
        self.bottom_length = []
        self.top_length = []
        # randomly generate height/width of all rectangles and make sure height!=width(range:30-80)
        while len(self.bottom_length) < self.num:
            bottom_len = random.randint(30, 80)
            top_len = random.randint(30, 80)
            if abs(bottom_len - top_len) > 20:
                self.bottom_length.append(bottom_len)
                self.top_length.append(top_len)
        self.height = np.random.randint(30, 80, size=self.num)
        self.shift = np.random.randint(-50, 50, size=self.num)
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'trapezoid'

    def draw_trapezoid(self, base_image, bottom_len, top_len, shift, height, color, center_width, center_height, angle):
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        rotation_matrix = cv2.getRotationMatrix2D((center_width, center_height), angle, 1)

        shape_points = np.array([
            [center_width, center_height],
            [center_width + bottom_len, center_height],
            [center_width + top_len + shift, center_height + height],
            [center_width + shift, center_height + height],
        ])
        rotated_points = cv2.transform(np.array([shape_points]), rotation_matrix)[0]
        rotated_points = rotated_points.reshape((-1, 1, 2))
        rotated_points = rotated_points.astype(int)
        return cv2.fillPoly(base_image, [rotated_points], color), cv2.fillPoly(mask_background, [rotated_points], color = (255, 255, 255) ) 

    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        angle = random.randint(0, 180)
        tag = [self.bottom_length[i], self.height[i], self.top_length[i], center_width, center_height, angle]
        drawn_image, mask = self.draw_trapezoid(base_image, self.bottom_length[i], self.top_length[i], self.shift[i],
                                          self.height[i], tuple(self.color[i].tolist()), center_width, center_height, angle)
        return drawn_image, mask, tag

class Parallelogram(Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/06_parallelogram"
            self.background_folder = "imgs_background/test/06_parallelogram"
            self.mask_folder = "imgs_mask/test/06_parallelogram"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/06_parallelogram"
            self.background_folder = "imgs_background/train/06_parallelogram"
            self.mask_folder = "imgs_mask/train/06_parallelogram"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        # randomly generate the radii and colors for all images
        self.length = np.random.randint(30, 80, size=self.num)
        self.height = np.random.randint(30, 80, size=self.num)
        self.shift = []
        for i in range(self.num):
            if np.random.rand() < 0.5:
                self.shift.append(np.random.randint(-60, -19))
            else:
                self.shift.append(np.random.randint(20, 61))
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'parallelogram'

    def draw_parallelogram(self, base_image, length, shift, height, color, center_width, center_height, angle):
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        rotation_matrix = cv2.getRotationMatrix2D((center_width, center_height), angle, 1)

        shape_points = np.array([
            [center_width, center_height],
            [center_width + length, center_height],
            [center_width + length + shift, center_height + height],
            [center_width + shift, center_height + height],
        ])
        rotated_points = cv2.transform(np.array([shape_points]), rotation_matrix)[0]
        rotated_points = rotated_points.reshape((-1, 1, 2))
        rotated_points = rotated_points.astype(int)
        return cv2.fillPoly(base_image, [rotated_points], color), cv2.fillPoly(mask_background, [rotated_points], color = (255, 255, 255) ) 

    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        angle = random.randint(0, 180)
        tag = [self.length[i], self.height[i], self.shift[i], center_width, center_height, angle]
        drawn_image, mask = self.draw_parallelogram(base_image, self.length[i], self.shift[i], self.height[i],
                                              tuple(self.color[i].tolist()), center_width, center_height, angle)
        return drawn_image, mask, tag

class Pentagon(Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/07_pentagon"
            self.background_folder = "imgs_background/test/07_pentagon"
            self.mask_folder = "imgs_mask/test/07_pentagon"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/07_pentagon"
            self.background_folder = "imgs_background/train/07_pentagon"
            self.mask_folder = "imgs_mask/train/07_pentagon"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        # randomly generate the radii and colors for all images
        self.radius = np.random.randint(30, 80, size=self.num)
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'pentagon'

    def draw_pentagon(self, base_image, radius, color, center_width, center_height, angle):
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        rotation_matrix = cv2.getRotationMatrix2D((center_width, center_height), angle, 1)
        shape_points = []
        for i in range(5):
            angle = 2 * math.pi * i / 5 - math.pi / 2  # 从正上方开始
            x = int(center_width + radius * math.cos(angle))
            y = int(center_height + radius * math.sin(angle))
            shape_points.append((x, y))
        rotated_points = cv2.transform(np.array([shape_points]), rotation_matrix)[0]
        rotated_points = rotated_points.reshape((-1, 1, 2))
        rotated_points = rotated_points.astype(int)
        return cv2.fillPoly(base_image, [rotated_points], color), cv2.fillPoly(mask_background, [rotated_points], color = (255, 255, 255) ) 

    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        angle = random.randint(0, 180)
        tag = [self.radius[i], center_width, center_height, angle]
        drawn_image, mask = self.draw_pentagon(base_image, self.radius[i], tuple(self.color[i].tolist()),
                                         center_width, center_height, angle)
        return drawn_image, mask, tag

class Hexagon(Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/08_hexagon"
            self.background_folder = "imgs_background/test/08_hexagon"
            self.mask_folder = "imgs_mask/test/08_hexagon"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/08_hexagon"
            self.background_folder = "imgs_background/train/08_hexagon"
            self.mask_folder = "imgs_mask/train/08_hexagon"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        # randomly generate the radii and colors for all images
        self.radius = np.random.randint(30, 80, size=self.num)
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'hexagon'

    def draw_hexagon(self, base_image, radius, color, center_width, center_height, angle):
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        rotation_matrix = cv2.getRotationMatrix2D((center_width, center_height), angle, 1)
        shape_points = []
        for i in range(6):
            angle = 2 * math.pi * i / 6 - math.pi / 2  # 从正上方开始
            x = int(center_width + radius * math.cos(angle))
            y = int(center_height + radius * math.sin(angle))
            shape_points.append((x, y))
        rotated_points = cv2.transform(np.array([shape_points]), rotation_matrix)[0]
        rotated_points = rotated_points.reshape((-1, 1, 2))
        rotated_points = rotated_points.astype(int)
        return cv2.fillPoly(base_image, [rotated_points], color), cv2.fillPoly(mask_background, [rotated_points], color = (255, 255, 255) ) 

    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        angle = random.randint(0, 180)
        tag = [self.radius[i], center_width, center_height, angle]
        drawn_image, mask = self.draw_hexagon(base_image, self.radius[i], tuple(self.color[i].tolist()),
                                        center_width, center_height, angle)
        return drawn_image, mask, tag

class Semicircle (Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/09_semicircle"
            self.background_folder = "imgs_background/test/09_semicircle"
            self.mask_folder = "imgs_mask/test/09_semicircle"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/09_semicircle"
            self.background_folder = "imgs_background/train/09_semicircle"
            self.mask_folder = "imgs_mask/train/09_semicircle"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        # randomly generate the radii and colors for all images
        self.radius = np.random.randint(30, 80, size=self.num)
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'semicircle '

    # draw the circle(since it makes no difference rotating the circle, we only need different center points)
    def draw_semicircle(self, base_image, radius, color, center_width, center_height, angle):
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        return cv2.ellipse(base_image, (center_width, center_height), (radius, radius),
                           angle=angle, startAngle=0, endAngle=180, color=color, thickness=-1),cv2.ellipse(mask_background, (center_width, center_height), (radius, radius),
                           angle=angle, startAngle=0, endAngle=180, color=(255,255,255), thickness=-1)



    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        angle = random.randint(0, 180)
        tag = [self.radius[i], center_width, center_height, angle]
        drawn_image, mask = self.draw_semicircle(base_image, self.radius[i], tuple(self.color[i].tolist()),
                                       center_width, center_height, angle)
        return drawn_image, mask, tag

class Ellipse (Geometry):
    def __init__(self, train):
        super().__init__()
        if train == False:
            self.output_folder = "imgs_main/test/10_ellipse"
            self.background_folder = "imgs_background/test/10_ellipse"
            self.mask_folder = "imgs_mask/test/10_ellipse"
            self.num = 50
        else:
            self.output_folder = "imgs_main/train/10_ellipse"
            self.background_folder = "imgs_background/train/10_ellipse"
            self.mask_folder = "imgs_mask/train/10_ellipse"
            self.num = 450
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.background_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        # randomly generate the radii and colors for all images
        self.radius_x = []
        self.radius_y = []
        # randomly generate height/width of all rectangles and make sure height!=width(range:30-80)
        while len(self.radius_x) < self.num:
            r_x = random.randint(30, 80)
            r_y = random.randint(30, 80)
            if abs(r_x - r_y) > 20:
                self.radius_x.append(r_x)
                self.radius_y.append(r_y)
        self.color = np.random.randint(0, 256, size=(self.num, 3), dtype=np.uint8)
        self.name = 'ellipse'

    # draw the circle(since it makes no difference rotating the circle, we only need different center points)
    def draw_ellipse(self, base_image, radius_x, radius_y, color, center_width, center_height, angle):
        mask_background = np.zeros((self.pic_size, self.pic_size, 3), dtype=np.uint8)
        return cv2.ellipse(base_image, (center_width, center_height), (radius_x, radius_y),
                           angle=angle, startAngle=0, endAngle=360, color=color, thickness=-1),cv2.ellipse(mask_background, (center_width, center_height), (radius_x, radius_y),
                           angle=angle, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)


    def draw_geometry(self, i):
        base_image = self.draw_background(i)
        center_width = random.randint(30, 210)
        center_height = random.randint(30, 210)
        angle = random.randint(0, 180)
        tag = [self.radius_x[i], self.radius_y[i], center_width, center_height, angle]
        drawn_image, mask = self.draw_ellipse(base_image, self.radius_x[i], self.radius_y[i], tuple(self.color[i].tolist()),
                                       center_width, center_height, angle)
        return drawn_image, mask, tag
# create instances of each class
rect = Rectangle(train=True)
rect.save_images()
rect = Rectangle(train=False)
rect.save_images()

sqr = Square(train=True)
sqr.save_images()
sqr = Square(train=False)
sqr.save_images()

cir = Circle(train=True)
cir.save_images()
cir = Circle(train=False)
cir.save_images()

tri = Triangle(train=True)
tri.save_images()
tri = Triangle(train=False)
tri.save_images()

trape = Trapezoid(train=True)
trape.save_images()
trape = Trapezoid(train=False)
trape.save_images()

para = Parallelogram(train=True)
para.save_images()
para = Parallelogram(train=False)
para.save_images()

penta = Pentagon(train=True)
penta.save_images()
penta = Pentagon(train=False)
penta.save_images()

hex = Hexagon(train=True)
hex.save_images()
hex = Hexagon(train=False)
hex.save_images()

semicir = Semicircle(train=True)
semicir.save_images()
semicir = Semicircle(train=False)
semicir.save_images()

ellip = Ellipse(train=True)
ellip.save_images()
ellip = Ellipse(train=False)
ellip.save_images()
# creating 3K images for each class and save



