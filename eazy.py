import numpy as np
import cv2
import open3d as o3d


import pykinect_azure as pykinect
from tqdm import tqdm

class Open3dVisualizer():
    def __init__(self, window_width=1600, window_height=1200):
        self.point_cloud = o3d.geometry.PointCloud()
        #self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size = 0.05)
        self.o3d_started = False
        #self.points_accumulated = []
        #self.colors_accumulated = []


        #Create a window with specified width and height
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=window_width, height=window_height)

    def __call__(self, points_3d):
        self.update(points_3d)

    def update(self, points_3d):
        #points_3d = np.asarray(points_3d)

        

        # #Filter points within 40cm (0.4m) away from the sensor
        # distance_threshold = 0.1
        # distances = np.linalg.norm(points_3d, axis=1)
        # points_within_threshold = points_3d[distances <= distance_threshold]

        # Update point cloud
        self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)

        
    
        # if rgb_image is not None:
        #     colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1, 3) / 255
        #     self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

        self.point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        if not self.o3d_started:
            self.vis.add_geometry(self.point_cloud)
            self.o3d_started = True
        else:
            self.vis.update_geometry(self.point_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()




# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

# Start device
device = pykinect.start_device(device_index=0,config=device_config) #R0 L1

# Initialize the Open3d visualizer
visualizer = Open3dVisualizer()


while True:
    # Get capture
    capture = device.update()

    # Get the 3D point cloud
    ret_point, points = capture.get_transformed_pointcloud()

    indices = (points[:,2] < 600)#right
        

    # Filter points and colors based on the condition
    points = points[indices]


    # Update visualizer
    visualizer.update(points)


