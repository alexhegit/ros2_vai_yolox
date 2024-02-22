# ros2_vai_yolox
------------------
A ROS2 node using YOLOX do inference with Vitis AI and ONNX runtime. You can deploy it with AMD Ryzen AI CPU to leveage the IPU to accelerate the inference.
This node subscribes the image msg from usb_cam and publish the inference image. And you can also modify it for other hardware EP of ONNX runtime.

## Prerequist

### Platform
1. Ryzen AI CPU Platform (e.g. Miniform Venus UM790 Pro/UM760 Pro with Ryzen 7940HS)
2. Ubuntu22.04
3. usb camera (e.g. logi-c270)

### Software
1. Setup Ryzen AI(VitisAI) software environment (refer to VitisAI documentattion to install the stack)
2. Install ROS2-humble (refer to https://docs.ros.org/en/humble/index.html)
3. Install ros2 usb_cam package (refer to https://index.ros.org/p/usb_cam/)
> sudo apt install ros-humble-usb-cam

## Build the ros2_vai_yolox package
1. source the ros2-humble environment
> source /opt/ros/humble/setup.sh
2. Clone the repo
> git clone https://gitenterprise.xilinx.com/alehe/ros2_vai_yolox.git

3. Build package
> The package include one node name /vai_yolox
> cd ros2_vai_yolox/ros2_vai_ws/
> colcon build

3. Check the ros2_vai_yolox package
> source install/local_setup.sh
> $ ros2 pkg list | grep vai_yolox
>> vai_yolox_pkg

## Run the demo
The demo pipeline is below. the /usb_cam node publish iamge_raw topic and /vai_yolox node subscribe the image_raw and publish the inference result as /image_infer topic.

**Steps**
The vai yolox inference depends on the IPU config file (in ros2_vai_ws/src/vai_yolox_pkg/resource/vaip_config.json) and yolox-s-int8.onnx. You should copy them to /tmp folder.(Now there are some hard codes to find them). The yolox-s-int8.onnx could be downloaded from https://huggingface.co/amd/yolox-s.

1. Open terminal 1 and run the /usb_cam
> ros2 run usb_cam usb_cam_node_exe

2. Open terminal 2 and run /vai_yolox
> ros2 run vai_yolox_pkg yolox_node

The yolox_node has many parameters with default values
```python
        self.declare_parameters(
            namespace = '',
            parameters = [
                ("model", "/tmp/yolox-s-int8.onnx"),
                ("vaip_config", "/tmp/vaip_config.json"),
                ("score_thr", 0.3),
                ("output_dir", "/tmp/demo_output"),
                ("image_path", "yolox-demo.jpg"),
                ("ipu", "True"),
                ("s_qos", 10, ),
                ("p_qos", 10)
            ]
        )
```
So you can run it with overide vaules for these parameters.
> e.g.
> 
> ros2 run vai_yolox_pkg yolox_node --ros-args -p model:=~/yolox-s-int8.onnx -p p_qos:=20

3. Open terminal 3 to check the pipeline graph
> ros2 run rqt_graph rqt_graph

4. Open terminal 4 to view the runtime inference video
> ros2 run rqt_image_view rqt_image_view

The whole demo looks like

