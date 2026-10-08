import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense devices detected. Check your connection.")
else:
    for d in devices:
        name = d.get_info(rs.camera_info.name)
        usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
        print(f"Device: {name}")
        print(f"Connection Type: {usb_type}")
        
        if "2." in usb_type:
            print("\nWARNING: Camera is connected via USB 2.x!")
            print("You must use a USB 3.0 port and a high-speed data cable.")