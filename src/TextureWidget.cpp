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

    updateImage();
    setFixedSize(width, height);
}

void TextureWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.drawPixmap(0, 0, pixmap);
}

void TextureWidget::keyPressEvent(QKeyEvent *event)
{
    // handle keydown events for translation and zooming
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
        update();
    }
    QWidget::keyPressEvent(event); 
}

void TextureWidget::updateImage()
{
    call_kernel(pixels, width, height, resolution, offsetX, offsetY);

    QImage image = QImage(reinterpret_cast<uchar*>(pixels), width, height, QImage::Format_ARGB32);
    pixmap = QPixmap::fromImage(image);
}
