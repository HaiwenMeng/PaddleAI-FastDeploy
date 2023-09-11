QT += gui

#CONFIG += c++11 console
CONFIG += c++11 console
CONFIG -= app_bundle

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
INCLUDEPATH+=\
     $$PWD/../include\

LIBS += \
    $$PWD/../lib/opencv_world460.lib\
    $$PWD/../lib/fastdeploy.lib\
    $$PWD/../lib/fastdeploy.lib\

HEADERS += \
    ytfastdeploylib.h\


SOURCES += \
        main.cpp\
        ytfastdeploylib.cpp\


INCLUDEPATH+=\
     $$PWD/../FastDeploy/build/include\


#DESTDIR+=F:\YtFastDeployLib\testFastDeploy\release



