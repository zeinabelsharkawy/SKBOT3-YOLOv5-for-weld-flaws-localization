# Enhanced-YOLOv5(SKBOT3-YOLOv5) for weld flaws localization
Developing YOLOv5 for weld flaws detection and localization in gamma radiography images based on attention mechanisms.
After image preprocessing,
1- You can use the official YOLOv5 from https://github.com/ultralytics/yolov5
2- Replace the common.py and yolo.py in the models folder.
3- Use WeldFlaws_data.yaml for data description and location. 
4- Use the proposed model in yolov5_SK_Bot.yaml.
5- Use yolov5s.pt, WeldFlaws_data.yaml ,and yolov5_SK_Bot.yaml in train.py.

Please, if you use this code, cite this paper " Elsharkawy, Z.F., Kasban, H. & Abbass, M.Y. Developing YOLOv5 for weld flaws detection and localization in gamma radiography images based on attention mechanisms. Multimed Tools Appl 85, 196 (2026). https://doi.org/10.1007/s11042-026-21272-z"
