import fiftyone
import numpy as np
from pyquaternion import Quaternion


class BBox3D:
    def __init__(self, solo_box):
        self.translation = solo_box.translation
        self.size = solo_box.size
        self.width, self.height, self.length = solo_box.size
        self.rotation = Quaternion(solo_box.rotation[3], solo_box.rotation[0], solo_box.rotation[1], solo_box.rotation[2])

    def _local2world_coordinate(self, pt):
        return np.array(self.translation) + self.rotation.rotate(pt)

    @property
    def back_left_bottom_pt(self):
        p = np.array([-self.width / 2, -self.height / 2, -self.length / 2])
        p = self._local2world_coordinate(p)
        return p

    @property
    def front_left_bottom_pt(self):
        p = np.array([-self.width / 2, -self.height / 2, self.length / 2])
        p = self._local2world_coordinate(p)
        return p

    @property
    def front_right_bottom_pt(self):
        p = np.array([self.width / 2, -self.height / 2, self.length / 2])
        p = self._local2world_coordinate(p)
        return p

    @property
    def back_right_bottom_pt(self):
        p = np.array([self.width / 2, -self.height / 2, -self.length / 2])
        p = self._local2world_coordinate(p)
        return p

    @property
    def back_left_top_pt(self):
        p = np.array([-self.width / 2, self.height / 2, -self.length / 2])
        p = self._local2world_coordinate(p)
        return p

    @property
    def front_left_top_pt(self):
        p = np.array([-self.width / 2, self.height / 2, self.length / 2])
        p = self._local2world_coordinate(p)
        return p

    @property
    def front_right_top_pt(self):
        p = np.array([self.width / 2, self.height / 2, self.length / 2])
        p = self._local2world_coordinate(p)
        return p

    @property
    def back_right_top_pt(self):
        p = np.array([self.width / 2, self.height / 2, -self.length / 2])
        p = self._local2world_coordinate(p)
        return p

    def _project_pt_to_pixel_location_perspective(self, pt, projection_matrix, img_dim):
        tmp = projection_matrix.dot(pt)

        # compute the perspective divide. Near clipping plane should take care of
        # divide by zero cases, but we will check to be sure
        if tmp[2] != 0:
            tmp /= tmp[2]

        return np.array(
            [
                int(-(tmp[0] * img_dim[0]) / 2.0 + (img_dim[0] * 0.5)),
                int((tmp[1] * img_dim[1]) / 2.0 + (img_dim[1] * 0.5)),
            ]
        )

    def _project_pt_to_pixel_location_ortho(self, pt, projection_matrix, img_dim):
        # The 'y' component needs to be flipped because of how Unity works
        projection = np.array(
            [
                [projection_matrix[0][0], 0, 0],
                [0, -projection_matrix[1][1], 0],
                [0, 0, projection_matrix[2][2]],
            ]
        )
        temp = projection.dot(pt)

        return [
            int((temp[0] + 1) * 0.5 * img_dim[0]),
            int((temp[1] + 1) * 0.5 * img_dim[1]),
        ]

    def to_polylines(self, img_dimension, camera_matrix, is_ortho = False):
        projected = self._project_to_image_coordinates(img_dimension, camera_matrix, is_ortho)

        return fiftyone.Polyline(
            points=[
                [projected[0], projected[1], projected[2], projected[3]],
                [projected[4], projected[5], projected[6], projected[7]],
                [projected[0], projected[4]],
                [projected[1], projected[5]],
                [projected[2], projected[6]],
                [projected[3], projected[7]],
            ],
            closed=True,
            filled=False
        )

    def _project_to_image_coordinates(self, img_dim, projection_matrix, ortho=False):
        bll = self.back_left_bottom_pt
        bul = self.back_left_top_pt
        bur = self.back_right_top_pt
        blr = self.back_right_bottom_pt

        fll = self.front_left_bottom_pt
        ful = self.front_left_top_pt
        fur = self.front_right_top_pt
        flr = self.front_right_bottom_pt

        projection_function = (
            self._project_pt_to_pixel_location_ortho
            if ortho
            else self._project_pt_to_pixel_location_perspective
        )

        fll_raster = projection_function(fll, projection_matrix, img_dim)
        ful_raster = projection_function(ful, projection_matrix, img_dim)
        fur_raster = projection_function(fur, projection_matrix, img_dim)
        flr_raster = projection_function(flr, projection_matrix, img_dim)
        bll_raster = projection_function(bll, projection_matrix, img_dim)
        bul_raster = projection_function(bul, projection_matrix, img_dim)
        bur_raster = projection_function(bur, projection_matrix, img_dim)
        blr_raster = projection_function(blr, projection_matrix, img_dim)

        w = img_dim[0]
        h = img_dim[1]

        return [
            (fll_raster[0] / w, fll_raster[1] / h), (ful_raster[0] / w, ful_raster[1] / h), (fur_raster[0] / w, fur_raster[1] / h), (flr_raster[0] / w, flr_raster[1] / h),
            (bll_raster[0] / w, bll_raster[1] / h), (bul_raster[0] / w, bul_raster[1] / h), (bur_raster[0] / w, bur_raster[1] / h), (blr_raster[0] / w, blr_raster[1] / h),
        ]
