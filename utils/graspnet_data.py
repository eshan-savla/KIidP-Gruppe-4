import glob
import os

from utils.dataset_processing import grasp, image
from utils.data.grasp_data import GraspDatasetBase


class GraspnetDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Graspnet dataset (reduced to 1 scene with only one groundtruth lable per image).
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Graspnet Scene 0 Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspnetDataset, self).__init__(**kwargs)


        self.grasp_files = glob.glob(os.path.join(file_path, 'graspnet_scene0/scenes/scene_0000/realsense/rect/', '0*.npy')) #  file_path passed as argument and equals: "data/"
        self.length = len(self.grasp_files)
        #print('grasp file length',self.length )

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = glob.glob(os.path.join(file_path, 'graspnet_scene0/scenes/scene_0000/realsense/depth/', '0*.png'))
        #print('depth files length', len(self.depth_files))
        self.rgb_files = glob.glob(os.path.join(file_path, 'graspnet_scene0/scenes/scene_0000/realsense/rgb/', '0*.png'))
        #print('rgb files length', len(self.rgb_files))

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_graspnet_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top#

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_graspnet_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs 

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_file(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
