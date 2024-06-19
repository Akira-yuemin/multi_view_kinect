import pcl
import pykinect_azure as pykinect
import pcl.pcl_visualization as viewer
import numpy as np
import cv2

pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

device = pykinect.start_device(config=device_config)
viewer = pcl.pcl_visualization.PCLVisualizering()

outputsize = (512, 512)

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
color_out = cv2.VideoWriter('F:/cuhk/MOIVEWORKSHOP/re/box_color_output9.avi', fourcc, 30.0, outputsize)
depth_out = cv2.VideoWriter('F:/cuhk/MOIVEWORKSHOP/re/box_depth_output9.avi', fourcc, 30.0, outputsize)  # Assuming the depth resolution is 1280x720
ir_out = cv2.VideoWriter('F:/cuhk/MOIVEWORKSHOP/re/box_ir_output9.avi', fourcc, 30.0, outputsize)  # Assuming the IR resolution is 1280x720q

while True:

    capture = device.update()

    # Get the color depth image from the capture
    ret, depth_image = capture.get_colored_depth_image()
    if not ret:
        continue

    # Get the color image from the capture
    ret, color_image = capture.get_transformed_color_image()
    if not ret:
        continue

    # Get the ir image from the capture
    ret, ir_image = capture.get_ir_image()
    if not ret:
        continue

    ret_point, points = capture.get_transformed_pointcloud()
    if ret_point is None:
        continue
    indices = (points[:, 2] < 700)  # right

    # Filter points and colors based on the condition
    points = points[indices]
    points[:, :2] = -points[:, :2]



    point_cloud = points
    point_cloud = np.array(point_cloud, dtype= np.float32)

    # ----------------------------------- #

    pcl_points = pcl.PointCloud(point_cloud)
    color1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(pcl_points, 0, 0, 255)
    viewer.AddPointCloud_ColorHandler(pcl_points, color1, b'scene_cloud', 0)

    # Plot the images
    color_image = cv2.flip(color_image[:, :, :3], 1)
    ir_image = cv2.flip(ir_image, 1)
    ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)



    depth_image = cv2.flip(depth_image, 1)

    cv2.imshow('Depth Image', depth_image)
    cv2.imshow('IR Image', ir_image)
    cv2.imshow('Color Image', color_image)

    viewer.SpinOnce()
    # viewer.Spin()

    viewer.RemovePointCloud(b'scene_cloud', 0)

    color_out.write(cv2.flip(color_image[:, :, :3], 0))
    depth_out.write(cv2.flip(depth_image, 0))
    ir_out.write(cv2.flip(ir_image, 0))

    if cv2.waitKey(1) == ord('q'):

        break
color_out.release()
depth_out.release()
ir_out.release()