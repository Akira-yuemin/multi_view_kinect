import pykinect_azure as pykinect
import numpy as np
import pandas as pd
import os


#way to process groups of data

def filesPath(path):

    filePaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root, file))
    return filePaths

def of_save(path):
    total = filesPath(path)
    with pd.ExcelWriter(f'{path}points.xlsx') as writer:
        i = 0
        for paths in total:
            arr = np.load(paths)
            DF = pd.DataFrame(arr, columns=['X', 'Y', 'Z'])
            DF.to_excel(writer, sheet_name=f'group{i}')
            print('processing:', i)
            i += 1


def mf_save(path):
    total = filesPath(path)
    i = 0
    for paths in total:
        arr = np.load(paths)
        zero_rows = np.all(arr[:, :3] == 0, axis=1)
        arr = arr[~zero_rows]

        DF = pd.DataFrame(arr)
        DF.to_csv(f'{path}{i:04d}.csv', header =None, index =None)
        print('processing:', i)
        i += 1


pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

device = pykinect.start_device(config=device_config)

index = 0
stack = []
path = 'F:/cuhk/MOIVEWORKSHOP/output/points/'

### total frames want to save
total_num = 20

while index <= total_num:


    capture = device.update()


    ret, points = capture.get_transformed_pointcloud()

    if not ret:
        continue

    #cv2.imwrite(f'F:/cuhk/MOIVEWORKSHOP/output/{index:5d}.png', depth_image)


    indices = (points[:, 2] < 700)  # distance boundary

    # Filter points and colors based on the condition
    points = points[indices]

    np.save(f'{path}{index:04d}.npy', points)

    index += 1

    print('processing:', index)

print('done npy saving')


# save to one file # too slow do not use
#of_save(path)

# save to multiple files  # too slow do not use
mf_save(path)

print('done csv transformation')





