#include "ytfastdeploylib.h"
#include <QDebug>
#include <QFile>
#include <QDir>
#include "yaml-cpp/yaml.h"

YtFastDeployLib::YtFastDeployLib()
{
    runtime_option_.UseCpu();
    m_ProMode=nullptr;
    detecte_model_ = nullptr;
    segment_model_=nullptr;
    class_model_=nullptr;
}

YtFastDeployLib::~YtFastDeployLib()
{
    if (detecte_model_ != nullptr)
    {
        delete detecte_model_;
    }
    if (segment_model_ != nullptr)
    {
        delete segment_model_;
    }
    if (class_model_ != nullptr)
    {
        delete class_model_;
    }
}



int YtFastDeployLib::toInitModel(QString ModeDir)
{

    QString model_file = ModeDir + "/inference.pdmodel";
    QString params_file = ModeDir + "/inference.pdiparams";
    QString config_file = ModeDir + "/inference.yml";

    if(m_ProMode!=nullptr)
    {
        delete m_ProMode;
        m_ProMode=nullptr;
    }

    //通过config判断模型类型
    YAML::Node cfg;
    try
    {
        cfg = YAML::LoadFile(config_file.toStdString().c_str());
    }
    catch (YAML::BadFile& e)
    {
        qDebug() << "Failed to load yaml file "
                << ", maybe you should check model files.";
        return kIllegalType;
    }

    if (cfg["arch"])
    {

//        qDebug() << QString("find arch in Det config, arch type:%1").arg(QString::fromStdString(cfg["arch"].as<std::string>())) ;
        SetModelType(kPaddleDetect);
    }
    else if(cfg["Deploy"])
    {
//        qDebug() << "find Seg config.";
        SetModelType(kPaddleSeg);
    }
    else if(cfg["PostProcess"])
    {
//        qDebug() << QString("find PostProcess in Cls config");
        SetModelType(kPaddleCls);
//        qDebug() << QString("set Cls config");
    }

    qDebug()<<"model_type_:"<<QString::number(model_type_);

    //初始化
    switch (model_type_)
    {
    case kPaddleDetect:
    {

        detecte_model_ = new fastdeploy::vision::detection::PaddleDetectionModel(model_file.toStdString().c_str(),\
                                                                                 params_file.toStdString().c_str(),\
                                                                                 config_file.toStdString().c_str(),\
                                                                                 runtime_option_);
        if (!detecte_model_->Initialized())
        {
//            qDebug() << "Failed to initialize DetectionModel." ;
            return kIllegalType;
        }
//        qDebug()<<"successfully initialized DetectionModel.";
        GetLabelList(ModeDir);
        m_ProMode=detecte_model_;
        return kPaddleDetect;

    }

    case kPaddleSeg:
    {

        segment_model_ = new fastdeploy::vision::segmentation::PaddleSegModel(model_file.toStdString().c_str(),\
                                                                              params_file.toStdString().c_str(),\
                                                                              config_file.toStdString().c_str(),\
                                                                              runtime_option_);
        if (!segment_model_->Initialized())
        {

//            qDebug() << "Failed to initialize SegModel.";
            return kIllegalType;
        }
//        qDebug()<<"successfully initialized SegModel";
        GetLabelList(ModeDir);
        m_ProMode=segment_model_;

        return kPaddleSeg;

    }
    case kPaddleCls:
    {
        qDebug() << "Start initialize classModel.";
        class_model_ = new fastdeploy::vision::classification::PaddleClasModel(model_file.toStdString().c_str(),\
                                                                               params_file.toStdString().c_str(),\
                                                                               config_file.toStdString().c_str(),\
                                                                               runtime_option_);
        if (!class_model_->Initialized())
        {
//            qDebug() << "Failed to initialize classModel.";
            return kIllegalType;

        }
//        qDebug() << "successfully initialized classModel.";
        GetLabelList(ModeDir);
        m_ProMode=class_model_;

        return kPaddleCls;
    }

    default:
        return kIllegalType;
    }

}

int YtFastDeployLib::toPredect(const QImage &Inim)
{

    if(model_type_ == kIllegalType)
    {
        return kIllegalType;
    }

    Qimage2cvMat(const_cast<QImage*>(&Inim), input_image_);
    if(input_image_.empty())
    {
        qDebug()<<"input image is empty.";
        return kIllegalType;
    }

    if(model_type_ == kPaddleDetect)
    {
        det_result_.Clear();
        if (detecte_model_->Predict(input_image_, &det_result_))
        {
            return kPaddleDetect;
        }
        else
        {
            return kIllegalType;
        }
    }

    if(model_type_ == kPaddleSeg)
    {
        seg_result_.Clear();
        if (segment_model_->Predict(input_image_, &seg_result_))
        {
            return kPaddleSeg;
        }
        else
        {
            return kIllegalType;
        }
    }

    if(model_type_ == kPaddleCls)
    {
        cls_result_.Clear();
        if (class_model_->Predict(input_image_, &cls_result_))
        {
            return kPaddleCls;
        }
        else
        {
            return kIllegalType;
        }
    }
}

void YtFastDeployLib::toGetDetcData(QVector<InferResult> &GetInferResult, double getPrsocre)
{
    if(model_type_ != kPaddleDetect)
    {
        qDebug()<<"can't run toGetDetcData() with other model type.";
        return;
    }

    GetInferResult.clear();

    std::vector<cv::Rect> nms_boxes;
    std::vector<int> nms_classIds;
    std::vector<float> nms_confidences;

    //NMS
    for(int i=0; i<det_result_.boxes.size();i++)
    {

        if (det_result_.scores[i] >= getPrsocre)
        {
            cv::Rect tem_box;
            tem_box.x = det_result_.boxes[i].at(0);
            tem_box.y = det_result_.boxes[i].at(1);
            tem_box.width = det_result_.boxes[i].at(2) - det_result_.boxes[i].at(0);
            tem_box.height = det_result_.boxes[i].at(3)- det_result_.boxes[i].at(1);

            nms_boxes.push_back(tem_box);
            nms_classIds.push_back(det_result_.label_ids[i]);
            nms_confidences.push_back(det_result_.scores[i]);
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(nms_boxes, nms_confidences,getPrsocre, nms_thresold_score, nms_result);
    if(nms_result.size()<1)
    {
        qDebug()<<"Inference success, not detect.";
//        return;
    }

    for(int i=0; i< nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        InferResult tem_infer_result;
        tem_infer_result.class_id = nms_classIds[idx];
        tem_infer_result.confidence = nms_confidences[idx];
        tem_infer_result.lefttop_x = nms_boxes[idx].x;
        tem_infer_result.lefttop_y = nms_boxes[idx].y;
        tem_infer_result.width    = nms_boxes[idx].width;
        tem_infer_result.height   = nms_boxes[idx].height;

        if(tem_infer_result.class_id<label_list_.size())
        {
            tem_infer_result.class_name = label_list_[tem_infer_result.class_id];
        }
        else
        {
            tem_infer_result.class_name = QString(u8"类别")+QString::number(tem_infer_result.class_id);
            qDebug()<<"class_id is lager than label_num, plz check labels.txt";
        }
        GetInferResult.append(tem_infer_result);
    }

}

void YtFastDeployLib::toGetClsData(InferResult &GetInferResult)
{
    if(model_type_ != kPaddleCls)
    {
        qDebug()<<"can't run toGetClsData() with other model type.";
        return;
    }
    qDebug()<<"cls_result_.scores.size: "<<cls_result_.scores.size();
    assert (cls_result_.scores.size()>0);
    auto max_score = std::max_element(cls_result_.scores.begin(), cls_result_.scores.end());
    int max_index;
    if (max_score != cls_result_.scores.end())
    {
        max_index = std::distance(cls_result_.scores.begin(), max_score);
    }
    GetInferResult.confidence = *max_score;
    GetInferResult.class_id = int(cls_result_.label_ids[max_index]);
    if(GetInferResult.class_id<label_list_.size())
    {
        GetInferResult.class_name = label_list_[GetInferResult.class_id];
    }
    else
    {
        GetInferResult.class_name = QString(u8"类别")+QString::number(GetInferResult.class_id);
        qDebug()<<"class_id is lager than label_num, plz check labels.txt";
    }

    GetInferResult.lefttop_x = 0;
    GetInferResult.lefttop_y = 0;
    GetInferResult.width    = 0;
    GetInferResult.height   = 0;

}

void YtFastDeployLib::toGetSegData(QImage &OutImage)
{
    if(model_type_ != kPaddleSeg)
    {
        qDebug()<<"can't run toGetSegData() with other model type.";
        return;
    }

    if(!input_image_.empty())
    {
        int mask_height = seg_result_.shape[0];
        int mask_width = seg_result_.shape[1];
        cv::Mat mask_png = cv::Mat(mask_height, mask_width, CV_8UC1);
        mask_png.data = seg_result_.label_map.data();
        for(int i=0; i<seg_result_.score_map.size(); i++)
        {
            if(seg_result_.score_map[i] < 0.5)
            {
                mask_png.data[i] = 0;
            }

        }
        cvMat2QImage(&mask_png, OutImage);
    }

}

QString YtFastDeployLib::toGetModelType()
{
    QStringList all_model_type;
    all_model_type<<u8"目标检测/Det"<<u8"分割/Seg"<<u8"分类/Cls";
    if(model_type_ != kIllegalType)
    {
        return all_model_type[model_type_];
    }
    else
    {
        return "NotInit";
    }
}

void YtFastDeployLib::SetModelType(int ModelType)
{
    model_type_ = ModelType;

    if(ModelType == kPaddleDetect)
    {
        if(class_model_ != nullptr)
        {
//            delete class_model_;
            class_model_ = nullptr;
        }
        if(segment_model_ != nullptr)
        {
//            delete segment_model_;
            segment_model_ = nullptr;
        }
    }
    else if(ModelType == kPaddleSeg)
    {
        if(class_model_ != nullptr)
        {
//            delete class_model_;
            class_model_ = nullptr;
        }
        if(detecte_model_ != nullptr)
        {
//            delete detecte_model_;
            detecte_model_ = nullptr;
        }

    }
    else if(ModelType == kPaddleCls)
    {
        segment_model_ = nullptr;
        detecte_model_ = nullptr;

//        if(segment_model_ != nullptr)
//        {
//            delete segment_model_;
//        }
//        if(detecte_model_ != nullptr)
//        {
//           delete detecte_model_;

//        }

    }
}

void YtFastDeployLib::GetLabelList(QString Label_Dir)
{
    if(model_type_ == kIllegalType)
    {
        qDebug()<<"plz init model before GetLabelList().";
        return;
    }
    QFile lable_file(Label_Dir+"/labels.txt");
    if(!lable_file.exists())
    {
        qDebug()<<QString("%1 not exist.").arg(Label_Dir+"/labels.txt");
        return;
    }
    bool is_Open = lable_file.open(QIODevice::ReadOnly);
    if(is_Open)
    {
        label_list_.clear();
        QTextStream label_stream(&lable_file);
        while(!label_stream.atEnd())
        {

            QString Line =QString::fromUtf8(label_stream.readLine().toUtf8());
            //            Line = Line.remove(QRegExp("\\s"));
            label_list_.append(Line);
        }
    }
}

void YtFastDeployLib::Qimage2cvMat(QImage* m_QImg, cv::Mat& m_Mat)
{
    qDebug()<<u8"Image format:"<<m_QImg->format();
    switch(m_QImg->format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
        *m_QImg = (QImage &)m_QImg->convertToFormat(QImage::Format_RGB888 );
        cv::Mat(m_QImg->height(), m_QImg->width(), CV_8UC3, (void*)m_QImg->constBits(), m_QImg->bytesPerLine()).copyTo(m_Mat);
        cv::cvtColor(m_Mat, m_Mat, cv::COLOR_BGR2RGB);

        break;
    case QImage::Format_ARGB32_Premultiplied:
        //param2为ptr,构造时不复制，用copyto()避免原始数据被操作
        cv::Mat(m_QImg->height(), m_QImg->width(), CV_8UC4, (void*)m_QImg->constBits(), m_QImg->bytesPerLine()).copyTo(m_Mat);
        break;
    case QImage::Format_RGB888:
        cv::Mat(m_QImg->height(), m_QImg->width(), CV_8UC3, (void*)m_QImg->constBits(), m_QImg->bytesPerLine()).copyTo(m_Mat);
        cv::cvtColor(m_Mat, m_Mat, cv::COLOR_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        cv::Mat(m_QImg->height(), m_QImg->width(), CV_8UC1, (void*)m_QImg->constBits(), m_QImg->bytesPerLine()).copyTo(m_Mat);
        break;
    case QImage::Format_RGBA64:
        cv::Mat mat(m_QImg->height(), m_QImg->width(), CV_16UC4, (void*)m_QImg->constBits(), m_QImg->bytesPerLine());
        cv::cvtColor(mat, m_Mat, cv::COLOR_BGRA2RGB);
        m_Mat.convertTo(m_Mat, CV_8UC3, 1.0 / 256.0);
        break;
    }

}

void YtFastDeployLib::cvMat2QImage(cv::Mat* m_Mat, QImage& m_QImg)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(m_Mat->type() == CV_8UC1)
    {
        m_QImg = QImage(m_Mat->cols, m_Mat->rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        m_QImg.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            m_QImg.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = m_Mat->data;
        for(int row = 0; row < m_Mat->rows; row ++)
        {
            uchar *pDest = m_QImg.scanLine(row);
            memcpy(pDest, pSrc, m_Mat->cols);
            pSrc += m_Mat->step;
        }
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(m_Mat->type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)m_Mat->data;
        // Create QImage with same dimensions as input Mat
        m_QImg = QImage(pSrc, m_Mat->cols, m_Mat->rows, m_Mat->step, QImage::Format_RGB888);
        m_QImg= m_QImg.rgbSwapped();
    }
    else if(m_Mat->type() == CV_8UC4)
    {
        qDebug() << "CV_8UC4";
        // Copy input Mat
        const uchar *pSrc = (const uchar*)m_Mat->data;
        // Create QImage with same dimensions as input Mat
        m_QImg = QImage(pSrc, m_Mat->cols, m_Mat->rows, m_Mat->step, QImage::Format_ARGB32);
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
    }

}
