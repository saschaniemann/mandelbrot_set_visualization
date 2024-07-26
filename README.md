# Mandelbrot Set Visualization
<!-- ![mandelbrot](https://github.com/user-attachments/assets/14c8b2fd-d206-42ff-9e8a-4276c700f075) -->
<img src="https://github.com/user-attachments/assets/14c8b2fd-d206-42ff-9e8a-4276c700f075" width="800" height="600" /> \
The Mandelbrot Set is a mathematical set. Its definition can be seen [here](https://en.wikipedia.org/wiki/Mandelbrot_set). 

## Requirements
We use Qt5 for the GUI.

## Installation and build
Install Qt5 (`sudo apt-get install qt5-default`). Use CMake to build the project as follows:
```
mkdir build
cd build
cmake ..
make
```

## Run the application
Run `./mandelbrot_set_visualization` in your terminal while still being in the build directory. \
You can use the arrow keys to move around and +/- to zoom in or out.
