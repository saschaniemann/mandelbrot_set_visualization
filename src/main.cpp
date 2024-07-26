#include <QApplication>
#include <QWidget>
#include "TextureWidget.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    TextureWidget widget(nullptr);
    widget.show();

    return app.exec();
}
