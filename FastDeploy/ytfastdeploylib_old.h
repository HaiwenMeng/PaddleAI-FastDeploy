#ifndef YTFASTDEPLOYLIB_H
#define YTFASTDEPLOYLIB_H
#include <fastdeploy/vision.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <QString>
#include <QImage>
#include <QVector>


////////
/*
depend path = $$PWD/../../../BaseLib$${BUILDPATH}/FastDeploy/bin\
*/

enum ModelType
{
    kIllegalType = -1,//类型错误
    kPaddleDetect,//目标识别
    kPaddleSeg,//语义分割
    kPaddleCls,//分类
};

struct InferResult
{
    int     class_id;
    QString class_name;
    double  confidence;
    double  lefttop_x;
    double  lefttop_y;
    double  width;
    double  height;
};


class YtFastDeployLib
{
public:
    YtFastDeployLib();
    ~YtFastDeployLib();

public:
    int toInitModel(QString ModeDir,int type=-1);
    //////////////
    int toPredect(const QImage &Inim);
//    int toPredect(char* data,int imagewid,int imagehig,int imchanle);
    void toGetDetcData(QVector<InferResult> &GetInferResult,double getPrsocre=0.5);
    void toGetClsData(InferResult &GetInferResult);
    void toGetSegData(QImage &OutImage);
    QString toGetModelType();


public:
    void Qimage2cvMat(QImage* m_QImg, cv::Mat& m_Mat);
    void cvMat2QImage(cv::Mat* m_Mat, QImage& m_QImg);

private:
    cv::Mat             input_image_;
    int                 model_type_{kIllegalType};
    QVector<QString>    label_list_;

    fastdeploy::vision::detection::PaddleDetectionModel     *detecte_model_{nullptr};
    fastdeploy::vision::segmentation::PaddleSegModel        *segment_model_{nullptr};
    fastdeploy::vision::classification::PaddleClasModel     *class_model_{nullptr};

    fastdeploy::RuntimeOption               runtime_option_;
    fastdeploy::vision::DetectionResult     det_result_;
    fastdeploy::vision::ClassifyResult      cls_result_;
    fastdeploy::vision::SegmentationResult  seg_result_;
    void SetModelType(int ModelType);
    void GetLabelList(QString Label_Dir);



};

#endif // YTFASTDEPLOYLIB_H
