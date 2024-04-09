from pyorbbecsdk import *
import cv2
import numpy as np
from utils import frame_to_bgr_image
import os

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

def save_depth_frame(frame: DepthFrame, index):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    timestamp = frame.get_timestamp()
    scale = frame.get_depth_scale()
    data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    data = data.reshape((height, width))
    
    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
    # Normalize the depth data to 0-255 range for visualization
    data = data.astype(np.float32) * scale  # Scale the depth to actual distances
    normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    save_image_dir = os.path.join(os.getcwd(), "depth_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    
    # Save the depth data as an 8-bit grayscale PNG image
    image_filename = save_image_dir + "/depth_{}x{}_{}_{}.png".format(width, height, index, timestamp)
    cv2.imwrite(image_filename, depth_image)



def save_color_frame(frame: ColorFrame, index):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    timestamp = frame.get_timestamp()
    save_image_dir = os.path.join(os.getcwd(), "color_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    filename = save_image_dir + "/color_{}x{}_{}_{}.png".format(width, height, index, timestamp)
    image = frame_to_bgr_image(frame)
    if image is None:
        print("failed to convert frame to image")
        return
    cv2.imwrite(filename, image)

def save_ir_frame(frame: InfraredFrame, index):
    if frame is None:
        return
    width = frame.get_width()
    height = frame.get_height()
    timestamp = frame.get_timestamp()
    ir_format = frame.get_format()
    if ir_format == OBFormat.Y8:
        ir_data = np.resize(ir_data, (height, width, 1))
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
    elif ir_format == OBFormat.MJPG:
        ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
        if ir_data is None:
            print("decode mjpeg failed")
            return
        ir_data = np.resize(ir_data, (height, width, 1))
    else:
        ir_data = np.frombuffer(ir_data, dtype=np.uint16)
        data_type = np.uint16
        image_dtype = cv2.CV_16UC1
        max_data = 65535
        ir_data = np.resize(ir_data, (height, width, 1))
    cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
    ir_data = ir_data.astype(data_type)
    ir_image = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)

    save_image_dir = os.path.join(os.getcwd(), "ir_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    filename = save_image_dir + "/color_{}x{}_{}_{}.png".format(width, height, index, timestamp)
    cv2.imwrite(filename, ir_image)



def main():
    pipeline = Pipeline()
    config = Config()
    saved_color_cnt: int = 0
    saved_depth_cnt: int = 0
    saved_ir_cnt: int = 0
    has_color_sensor = False
    has_ir_sensor = False
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is not None:
            color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            has_color_sensor = True
    except OBError as e:
        print(e)
    try:
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is not None:
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
    except OBError as e:
        print(e)
    try:
        ir_profile_list = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
        if ir_profile_list is not None:
                ir_profile = ir_profile_list.get_default_video_stream_profile()
                config.enable_stream(ir_profile)
                has_ir_sensor = True
    except OBError as e:
        print(e)
    pipeline.start(config)
    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            if has_color_sensor:
                if has_ir_sensor:
                    if saved_color_cnt >= 5 and saved_depth_cnt >= 5 and saved_ir_cnt >= 5:
                        break
                else:
                    if saved_color_cnt >= 5 and saved_depth_cnt >= 5:
                        break
            elif has_ir_sensor:
                if saved_ir_cnt >= 5 and saved_depth_cnt >= 5:
                        break
            color_frame = frames.get_color_frame()
            if color_frame is not None and saved_color_cnt < 5:
                save_color_frame(color_frame, saved_color_cnt)
                saved_color_cnt += 1
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None and saved_depth_cnt < 5:
                save_depth_frame(depth_frame, saved_depth_cnt)
                saved_depth_cnt += 1
            ir_frame = frames.get_ir_frame()
            if ir_frame is not None and saved_ir_cnt < 5:
                save_ir_frame(ir_frame, saved_ir_cnt)
                saved_ir_cnt += 1

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()