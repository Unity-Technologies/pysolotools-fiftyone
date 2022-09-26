import argparse
import json
import os
import os.path
import pathlib
import sys

import fiftyone
import fiftyone.utils.data
import Imath
import numpy as np
import numpy.random
import OpenEXR as exr
import PIL.Image
from pysolotools_fiftyone.exr_utils import convert_exr_to_png
from pyquaternion import Quaternion
from pysolotools.consumers.solo import Solo
from pysolotools.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox3DAnnotation,
    DepthAnnotation,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    NormalAnnotation,
    PixelPositionAnnotation,
    RGBCameraCapture,
    SemanticSegmentationAnnotation,
)

from pysolotools_fiftyone.bounding_box_3d import BBox3D

BBOX_KEY = "unity.solo.BoundingBox2DAnnotation"
BBOX3D_KEY = "unity.solo.BoundingBox3DAnnotation"
KEYPOINT_KEY = "unity.solo.KeypointAnnotation"
INSTANCE_KEY = "unity.solo.InstanceSegmentationAnnotation"
METADATA_KEY = "type.unity.com/unity.solo.MetadataMetric"
RENDERED_OBJECT_INFO_KEY = "type.unity.com/unity.solo.RenderedObjectInfoMetric"
SEMANTIC_KEY = "unity.solo.SemanticSegmentationAnnotation"
PIXEL_POSITION_KEY = "unity.solo.PixelPositionAnnotation"
DEPTH_KEY = "unity.solo.DepthAnnotation"
OCCLUSION_KEY = "unity.solo.OcclusionMetric"
NORMAL_KEY = "unity.solo.NormalAnnotation"


class SoloDatasetImporter(fiftyone.utils.data.GroupDatasetImporter):
    """Class used to import solo data into fiftyone."""

    @property
    def has_sample_field_schema(self):
        return False

    def __init__(self, dataset_dir, shuffle=False, seed=None, max_samples=None):
        """Initializer.

        Parameters
        ----------
        dataset_dir: str
            The dataset directory
        shuffle: bool
            Should the frames be shuffled
        seed: int
            Random seed
        max_samples: int
            The max number of samples to load
        """
        super().__init__(
            dataset_dir=dataset_dir, shuffle=shuffle, seed=seed, max_samples=max_samples
        )

        self._solo_dir = dataset_dir
        self._labels_file = None
        self._labels = None
        self._iter_labels = None
        self._solo = Solo(dataset_dir)
        self.__frames_cache = None
        self._active_annotations = self._check_for_annotations(self._solo)
        self.skeletons = {}
        print(self._active_annotations)

    @staticmethod
    def _metadata_contains_annotator(annotators, label):
        return any(label in x["type"] for x in annotators)

    def _check_for_annotations(self, solo):
        ann_map = {}

        annotators = solo.metadata.annotators
        ann_map[KEYPOINT_KEY] = self._metadata_contains_annotator(
            annotators, KEYPOINT_KEY
        )
        ann_map[BBOX_KEY] = self._metadata_contains_annotator(annotators, BBOX_KEY)
        ann_map[BBOX3D_KEY] = self._metadata_contains_annotator(annotators, BBOX3D_KEY)
        ann_map[INSTANCE_KEY] = self._metadata_contains_annotator(
            annotators, INSTANCE_KEY
        )
        ann_map[SEMANTIC_KEY] = self._metadata_contains_annotator(
            annotators, SEMANTIC_KEY
        )
        ann_map[DEPTH_KEY] = self._metadata_contains_annotator(annotators, DEPTH_KEY)
        ann_map[PIXEL_POSITION_KEY] = self._metadata_contains_annotator(annotators, PIXEL_POSITION_KEY)
        ann_map[NORMAL_KEY] = self._metadata_contains_annotator(annotators, NORMAL_KEY)


        # TODO: This is checking for the name of the labeler, but it should be checking the type. The type isn't provided!
        ann_map[METADATA_KEY] = "metadata" in solo.metadata.metricCollectors

        return ann_map

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._solo)

    @property
    def frames(self):
        if not self.__frames_cache:
            self.__frames_cache = self._solo.frames()
        return self.__frames_cache

    @property
    def has_dataset_info(self):
        return True

    def get_dataset_info(self):
        dataset_info = {"skeletons": {}}

        if ann_def := self._get_annotation_definitions():
            for i in ann_def["annotationDefinitions"]:
                if i["@type"] == "type.unity.com/unity.solo.KeypointAnnotation":
                    template = i["template"]
                    if template is not None:
                        keypoints = template["keypoints"]

                        skeleton_edges = []
                        labels = [keypoint["label"] for keypoint in keypoints]

                        if "skeleton" in template:
                            skeleton = template["skeleton"]
                            for bone in skeleton:
                                skeleton_edges.append([bone["joint1"], bone["joint2"]])

                        dataset_info["skeletons"][
                            "keypoints"
                        ] = fiftyone.KeypointSkeleton(
                            labels=labels, edges=skeleton_edges
                        )

        return dataset_info

    @property
    def label_cls(self):
        return {
            "keypoints": fiftyone.Keypoints,
            "bbox": fiftyone.Detections,
            "semantic": fiftyone.Segmentation,
            "instance": fiftyone.Segmentation,
            "depth": fiftyone.Heatmap,
        }

    @property
    def has_image_metadata(self):
        return True

    def _to_fiftyone_instance_segmentation(self, rgb, annotation, sequence, step):
        path = f"{self.dataset_dir}/sequence.{sequence}/step{step}.{rgb.id}.{annotation.id}.png"
        return self._to_fiftyone_segmentation(path), path

    def _to_fiftyone_semantic_segmentation(self, rgb, annotation, sequence, step):
        path = f"{self.dataset_dir}/sequence.{sequence}/step{step}.{rgb.id}.{annotation.id}.png"
        return self._to_fiftyone_segmentation(path), path

    @staticmethod
    def _to_fiftyone_segmentation(path):
        if not os.path.exists(path):
            return

        img = PIL.Image.open(path)

        # Convert to paletized image. The palette (int -> color mapping) can be accessed via img.get_palette()
        img = img.convert("P")

        n_img = numpy.asarray(img).astype(np.uint8)

        return n_img

    @staticmethod
    def _to_fiftyone_bbox(rgb, boxes, detections2d):
        out_boxes = []
        width, height = rgb.dimension
        for box in boxes.values:
            detection = detections2d.get(box.instanceId)
            if detection is None:
                detection = fiftyone.Detection()
                detections2d[box.instanceId] = detection

            ul = box.origin[0] / width
            ur = box.origin[1] / height
            w = box.dimension[0] / width
            h = box.dimension[1] / height

            detection.label = box.labelName
            detection.bounding_box = [ul, ur, w, h]
            out_boxes.append(detection)

        return out_boxes

    def _get_annotation_definitions(self):
        ann_def = None
        with open(f"{self.dataset_dir}/annotation_definitions.json", "r") as f:
            ann_def = json.load(f)
        return ann_def

    def _get_template(self, template_id):
        if ann_def := self._get_annotation_definitions():
            for i in ann_def["annotationDefinitions"]:
                if i["@type"] == "type.unity.com/unity.solo.KeypointAnnotation":
                    template = i["template"]
                    if template["templateId"] == template_id:
                        return template
        return None

    def _instances_to_palette(self, instance_annotation: InstanceSegmentationAnnotation):

        palette_data = np.full(256 * 3, 0, dtype=np.uint8)
        id_to_index_map = {}
        curr = 0

        instances = instance_annotation.instances

        if len(instances) > 256:
            print(f"Occlusion heatmap only supports up to 256 unique instances, received: {len(instances)}")
            return palette_data

        for i in instances:
            id_to_index_map[i.instanceId] = curr
            idx = curr * 3

            palette_data[idx: idx + 3] = i.color[0:3]
            curr += 1

        return palette_data, id_to_index_map

    def _create_palettized_image(self, palette):
        p_img = PIL.Image.new('P', (16, 16))
        p_img.putpalette(palette)
        return p_img

    def _get_occlusion_map(self, occlusion_metric, id_to_index_map):
        values = occlusion_metric["values"]
        occlusion_map = np.zeros(len(values) + 1, dtype=float)

        for v in values:
            idx = id_to_index_map[v["instanceId"]]
            value = v["percentVisible"]
            occlusion_map[idx] = value

        return occlusion_map

    def _occlusion_to_fiftyone_heatmap(self, occlusion_metric, rgb, annotation, sequence, step):
        # get the instance segmentation file

        instance_path = f"{self.dataset_dir}/sequence.{sequence}/step{step}.{rgb.id}.{annotation.id}.png"
        instance_img = PIL.Image.open(instance_path)

        palette, id_to_index_map = self._instances_to_palette(annotation)

        pal_img = self._create_palettized_image(palette)

        instance_img.load()
        pal_img.load()
        new_img = instance_img.im.convert("P", 0, pal_img.im)
        new_img_array = np.asarray(new_img)

        occlusion_map = self._get_occlusion_map(occlusion_metric, id_to_index_map)

        heatmap = occlusion_map[new_img_array]

        heatmap = heatmap.reshape(instance_img.height, instance_img.width)

        return fiftyone.Heatmap(map=heatmap)

    def _normal_or_pixel_position_to_group(self, frame, metadata):
        path = f"{self.dataset_dir}/sequence.{frame.sequence}/{metadata.filename}"
        exr_path = pathlib.Path(path)

        dir_name = exr_path.parent
        stem = exr_path.stem

        png_path = pathlib.Path(f"{self.dataset_dir}/fiftyone_images/sequence.{frame.sequence}/")

        png_path.mkdir(parents=True, exist_ok=True)

        png_path = png_path / f'{stem}.png'

        # check if we already have a cached converted png version
        cached_png = pathlib.Path.exists(png_path)

        if not cached_png:
            print(f'no cached png')
            convert_exr_to_png(str(exr_path), str(png_path))
        else:
            print(f'found cached png: {png_path}')

        return str(png_path)

    def _depth_to_fifytone_heatmap(self, frame, metadata):

        path = f"{self.dataset_dir}/sequence.{frame.sequence}/{metadata.filename}"
        exrfile = exr.InputFile(path)
        header = exrfile.header()
        dw = header['dataWindow']
        isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        channelData = dict()

        for c in header['channels']:
            channel = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
            channel = np.frombuffer(channel, dtype=np.float32)
            channel = np.reshape(channel, isize)

            channelData[c] = channel

        img = channelData['R']

        img = img / np.max(img)

        return fiftyone.Heatmap(map=img)

    def _to_fiftyone_bbox3d(self, rgb, boxes):
        out_boxes = []
        ortho = rgb.projection == "Orthographic"
        matrix = np.array(
            [
                [rgb.matrix[0], rgb.matrix[1], rgb.matrix[2]],
                [rgb.matrix[3], rgb.matrix[4], rgb.matrix[5]],
                [rgb.matrix[6], rgb.matrix[7], rgb.matrix[8]],
            ]
        )

        for box in boxes.values:
            b3d = BBox3D(box)
            out_boxes.append(b3d.to_polylines(rgb.dimension, matrix, ortho))

        return fiftyone.Polylines(polylines=out_boxes)

    def _to_fiftyone_keypoints(self, rgb, kps):
        out_kps = []
        out_skeletons = []

        width, height = rgb.dimension

        template_id = kps.templateId
        template = self._get_template(template_id)

        for figure in kps.values:
            # get the template

            if template is not None and (skeleton := template.get("skeleton", None)):
                skeleton_pts = []
                for bone in skeleton:
                    j1 = figure.keypoints[bone["joint1"]]
                    j2 = figure.keypoints[bone["joint2"]]

                    if j1.state == 2 and j2.state == 2:
                        x1 = j1.location[0] / width
                        y1 = j1.location[1] / height
                        x2 = j2.location[0] / width
                        y2 = j2.location[1] / height
                        skeleton_pts.append([(x1, y1), (x2, y2)])

                out_skeletons.append(fiftyone.Polyline(label="2", points=skeleton_pts))

            ps = []
            points = figure.keypoints
            for p in points:
                if p.state == 2:
                    ps.append((p.location[0] / width, p.location[1] / height))
                else:
                    ps.append((float("nan"), float("nan")))

            out_kps.append(fiftyone.Keypoint(label=f"{figure.instanceId}", points=ps))

        return out_kps, out_skeletons

    def _read_metadata(self, annotation, metadata, detections2d):
        values = annotation["values"]
        if (not bool(values)):
            return
        first = values[0]

        if not isinstance(first, dict) or "instances" not in first:
            return

        instances = first["instances"]
        for instance in instances:
            instance_id = instance["instanceId"]
            detection = detections2d.get(int(instance_id))
            if detection is None:
                detection = fiftyone.Detection()
                detections2d[instance_id] = detection

            for key in iter(instance):
                if key == "instanceId":
                    continue

                detection[key] = json.dumps(instance[key])

    def _read_rendered_object_info(self, annotation, metadata, detections2d):
        for value in annotation["values"]:
            instance_id = value["instanceId"]
            detection = detections2d.get(int(instance_id))
            if detection is None:
                detection = fiftyone.Detection()
                detections2d[instance_id] = detection

            for key in value:
                detection[key] = value[key]

    def _get_metric_for_capture(self, frame, metric_key, sensor_name):
        for metric in frame.metrics:
            if metric["@type"] == f'type.unity.com/{metric_key}':
                if metric["sensorId"] == sensor_name:
                    return metric

        return None

    def __next__(self):
        curr = self.frames.__next__()

        img_path = None
        metadata = None

        detections = {}

        group = fiftyone.Group()

        for sensor in curr.captures:

            # for right now only support the first RGB camera
            if isinstance(sensor, RGBCameraCapture):
                sensor_id = sensor.id
                img_path = f"{self.dataset_dir}/sequence.{curr.sequence}/step{curr.step}.{sensor_id}.png"
                metadata = fiftyone.ImageMetadata.build_for(img_path)
                normal_path = None
                pixel_position_path = None

                detections2d = {}

                for metric in curr.metrics:
                    if metric["sensorId"] != '' and metric["sensorId"] != sensor_id:
                        continue

                    if metric["@type"] == METADATA_KEY:
                        self._read_metadata(metric, metadata, detections2d)

                    if metric["@type"] == RENDERED_OBJECT_INFO_KEY:
                        self._read_rendered_object_info(metric, metadata, detections2d)

                for annotation in sensor.annotations:

                    if self._active_annotations[SEMANTIC_KEY] and isinstance(
                            annotation, SemanticSegmentationAnnotation
                    ):
                        mask, _ = self._to_fiftyone_semantic_segmentation(
                                sensor, annotation, curr.sequence, curr.step
                            )

                        detections["semantic"] = fiftyone.Segmentation(mask=mask)

                    if self._active_annotations[INSTANCE_KEY] and isinstance(
                            annotation, InstanceSegmentationAnnotation
                    ):
                        mask, _ = self._to_fiftyone_instance_segmentation(
                            sensor, annotation, curr.sequence, curr.step
                        )

                        detections["instance"] = fiftyone.Segmentation(mask=mask)

                    if self._active_annotations[BBOX_KEY] and isinstance(
                            annotation, BoundingBox2DAnnotation
                    ):
                        detections["bbox"] = fiftyone.Detections(
                            detections=self._to_fiftyone_bbox(sensor, annotation, detections2d)
                        )

                    if self._active_annotations[KEYPOINT_KEY] and isinstance(
                            annotation, KeypointAnnotation
                    ):
                        pts, skeleton = self._to_fiftyone_keypoints(sensor, annotation)
                        d = fiftyone.Keypoints(keypoints=pts)
                        detections["keypoints"] = d
                        e = fiftyone.Polylines(polylines=skeleton)
                        detections["skeleton"] = e

                    if self._active_annotations[DEPTH_KEY] and isinstance(annotation, DepthAnnotation):
                        detections["depth"] = self._depth_to_fifytone_heatmap(curr, annotation)
                        pass

                    if self._active_annotations[NORMAL_KEY] and isinstance(annotation, NormalAnnotation):
                        normal_path = self._normal_or_pixel_position_to_group(curr, annotation)

                    if self._active_annotations[PIXEL_POSITION_KEY] and isinstance(annotation, PixelPositionAnnotation):
                        pixel_position_path = self._normal_or_pixel_position_to_group(curr, annotation)

                    if self._active_annotations[BBOX3D_KEY] and isinstance(annotation, BoundingBox3DAnnotation):
                        detections["bbox3D"] = self._to_fiftyone_bbox3d(sensor, annotation)

                    if self._active_annotations[INSTANCE_KEY] and isinstance(annotation,
                                                                             InstanceSegmentationAnnotation):
                        metric = self._get_metric_for_capture(curr, OCCLUSION_KEY, sensor_id)

                        if metric:
                            detections["occlusion"] = self._occlusion_to_fiftyone_heatmap(metric, sensor, annotation,
                                                                                          curr.sequence, curr.step)

        sample = fiftyone.Sample(filepath=img_path, group=group.element("rgb"), metadata=metadata)
        sample.add_labels(detections)

        groups = {"rgb": sample}

        if normal_path is not None:
            s = fiftyone.Sample(filepath=normal_path, group=group.element("normals"))
            s.add_labels(detections)
            groups["normals"] = s

        if pixel_position_path is not None:
            s = fiftyone.Sample(filepath=pixel_position_path, group=group.element("pixel position"))
            s.add_labels(detections)
            groups["pixel position"] = s


        return groups
        # return img_path, metadata, detections

    def setup(self):
        pass

    def close(self, *args):
        pass


class SessionManager:
    def __init__(self, solo):
        self._session = None
        self._solo = solo

    def _start_internal(self):
        name = "solo_dataset"
        if fiftyone.dataset_exists(name):
            fiftyone.delete_dataset(name)

        dataset = fiftyone.Dataset()
        dataset.name = name
        dataset.persistent = True

        dataset.add_group_field("group", default="rgb")

        self._session = fiftyone.launch_app(dataset, auto=False)

        importer = SoloDatasetImporter(self._solo, max_samples=5)
        dataset.add_importer(importer)

    def start(self):
        self._start_internal()
        self._session.wait()

    def start_in_notebook(self):
        self._start_internal()
        self._session.show()


def run_in_notebook(solo):
    session = SessionManager(solo)
    session.start_in_notebook()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("solo")
    args = parser.parse_args(args)

    session = SessionManager(args.solo)
    session.start()


def cli():
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
