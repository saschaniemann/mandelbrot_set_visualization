#ifndef TEXTUREWIDGET_H
#define TEXTUREWIDGET_H

#include <QWidget>
#include <QPixmap>

class TextureWidget : public QWidget
{
    Q_OBJECT

    public:
        TextureWidget(QWidget *parent = nullptr);

    protected:
        void paintEvent(QPaintEvent *event) override;
        void keyPressEvent(QKeyEvent *event) override;

    private:
        void updateImage();

        uint32_t *pixels;
        QPixmap pixmap;
        int width;
        int height;
        float offsetX;
        float offsetY;
        float resolution;
};

#endif // TEXTUREWIDGET_H
