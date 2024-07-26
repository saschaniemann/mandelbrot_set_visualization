#include "TextureWidget.h"
#include <QPainter>
#include <QKeyEvent>
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include<chrono>
#include<iostream>

TextureWidget::TextureWidget(QWidget *parent) : QWidget(parent) {
    width = 1000;
    height = 800;
    offsetX = 0.5;
    offsetY = 0;
    resolution = 2.5;

    pixels = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * width * height));

    // Initialize QImage with the image data
    updateImage();
    // Set the preferred size of the widget to the size of the image
    setFixedSize(width, height);
}

void TextureWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.drawImage(0, 0, image);
}

void TextureWidget::keyPressEvent(QKeyEvent *event)
{
    bool keyDown = true;
    if (event->key() == Qt::Key_Up) {
        offsetY += 0.1 * resolution;
    }
    else if (event->key() == Qt::Key_Down) {
        offsetY -= 0.1 * resolution;
    }
    else if (event->key() == Qt::Key_Left) {
        offsetX += 0.1 * resolution;
    }
    else if (event->key() == Qt::Key_Right) {
        offsetX -= 0.1 * resolution;
    }
    else if (event->key() == Qt::Key_Minus) {
        resolution *= 1.5;
    }
    else if (event->key() == Qt::Key_Plus) {
        resolution *= 0.7;
    }
    else {
        keyDown = false;
    }
    if(keyDown) {
        updateImage();
        std::chrono::steady_clock::time_point beginCallUpdate = std::chrono::steady_clock::now();
        update(); // Request a repaint
        std::chrono::steady_clock::time_point endCallUpdate = std::chrono::steady_clock::now();
        std::cout << "Time for updating image: " << std::chrono::duration_cast<std::chrono::milliseconds>(endCallUpdate - beginCallUpdate).count() << "[ms]" << std::endl;
    }
    QWidget::keyPressEvent(event); // Call base class implementation
}

void TextureWidget::updateImage()
{
    std::chrono::steady_clock::time_point beginCallKernel = std::chrono::steady_clock::now();
    call_kernel(pixels, width, height, resolution, offsetX, offsetY);
    std::chrono::steady_clock::time_point endCallKernel = std::chrono::steady_clock::now();
    std::cout << "Time for running call_kernel: " << std::chrono::duration_cast<std::chrono::milliseconds>(endCallKernel - beginCallKernel).count() << "[ms]" << std::endl;

    image = QImage(reinterpret_cast<uchar*>(pixels), width, height, QImage::Format_ARGB32);
    endCallKernel = std::chrono::steady_clock::now();
    std::cout << "Time for running updateImage(): " << std::chrono::duration_cast<std::chrono::milliseconds>(endCallKernel - beginCallKernel).count() << "[ms]" << std::endl;
}
