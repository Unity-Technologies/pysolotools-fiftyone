# Solo Voxel51 Viewer
## Requirements
Make sure that you are in the **solo_fiftyone** directory
```shell
cd pysolo-extensions/viewers/solo_fiftyone
```
Solo Voxel51 viewer requires:
- pysolotools
- Voxel51
```shell
# make sure that you are in the solo_fiftyone directory
pip install -r requirements.txt
```

## Running
Make sure that you are in the **solo_fiftyone** directory
```shell
cd pysolo-extensions/viewers/solo_fiftyone
```
The Solo Fiftyone viewer is executed on the command line with the following command line:
```shell
python solo_fiftyone.py ./examples/solo_10_all_labelers
```

This will launch a new fiftyone viewer in your web browser.

## Annotations Supported
Currently the solo Voxel51 viewer support the following the annotations:
- Bounding Box 2D
- Keypoints
- Instance Segmentation
- Semantic Segmentation

## Know Issues
When running with a web browser, it will open with a subset of your frames visible.
The data is still being imported in the background, and refreshing the browser will
update the viewer will the current set of loaded scenes.
