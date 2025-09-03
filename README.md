# :hand: Hand Gesture Website Launcher

This project uses hand tracking and gesture recognition to open websites through simple hand movements. Instead of clicking or typing, you can assign specific hand gestures to launch your favorite websites instantly.

Currently in the testing stage, this project was built as a personal experiment in computer vision and human-computer interaction, showing how intuitive and touchless control can be applied to everyday browsing.

## Installation
Only runs in Python 3.11.x
```
py -3.11 pip install mediapipe opencv-python mediapipe pickle numpy
```

## Usage
First, you must implement your own data of hand signs. In the **collect-data.py**, you may change the amount of hand gestures and photos of each (improving the ai)
```
num_classes = #
dataset_size = # (Suggested 200-300)
```
Whenever you are ready, Run **collect-data.py** then press the letter q on your keyboard to collect the data (while its running I suggest moving the hand gestures further away from the camera) until the poptext tells you to press it again. Afterwards it will automatically close letting you know that the all the data is collected.
