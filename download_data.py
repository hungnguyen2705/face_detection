from roboflow import Roboflow
rf = Roboflow(api_key="sS90mJHxZ9r2xJJ3w6u4")
project = rf.workspace("face-detection-0illn").project("face-detection-d2hlf")
version = project.version(4)
dataset = version.download("yolov8")
                