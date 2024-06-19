import cv2
import pykinect_azure as pykinect


if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    # Define the codec and create VideoWriter objects
    outputsize = (512, 512)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    color_out = cv2.VideoWriter('F:\cuhk\MOIVEWORKSHOP/re/box_color_output9.avi', fourcc, 30.0, outputsize)
    depth_out = cv2.VideoWriter('F:\cuhk\MOIVEWORKSHOP/re/box_depth_output9.avi', fourcc, 30.0,
                                outputsize)  # Assuming the depth resolution is 1280x720
    ir_out = cv2.VideoWriter('F:\cuhk\MOIVEWORKSHOP/re/box_ir_output9.avi', fourcc, 30.0,
                             outputsize)  # Assuming the IR resolution is 1280x720
    test_out = cv2.VideoWriter('F:\cuhk\MOIVEWORKSHOP/re/box_ir_test9.avi', fourcc, 30.0,
                               outputsize)  # Assuming the IR resolution is 1280x720

    # cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('IR Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Color Image', cv2.WINDOW_NORMAL)
    frame_num = 0

    while True:
        # Get capture
        capture = device.update()

        ret_point, points = capture.get_transformed_pointcloud()

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

        # Plot the images
        cv2.imshow('Depth Image', depth_image)
        cv2.imshow('IR Image', ir_image)
        cv2.imshow('Color Image', color_image)

        ir_image = ir_image.astype('uint8')
        # ret, ir_image = cv2.threshold(ir_image, 120, 255, cv2.THRESH_BINARY)
        ir_image = cv2.blur(ir_image, (1, 1))
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
        color_image = cv2.blur(color_image, (1, 1))
        depth_image = cv2.blur(depth_image, (1, 1))



        #add weight fuction as you like
        test_image = cv2.addWeighted(color_image[:, :, :3], 0.7, depth_image, 0.3, 0)

        # Write the frames to the video files
        color_out.write(color_image[:, :, :3])
        depth_out.write(depth_image)
        ir_out.write(ir_image)
        test_out.write(test_image)

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            print(color_image)
            break

    # Release everything if job is finished
    color_out.release()
    depth_out.release()
    ir_out.release()
    test_out.release()

    cv2.destroyAllWindows()
