#include "ytfastdeploylib.h"
#include <QDebug>
#include <QFile>
#include <QDir>

YtFastDeployLib::YtFastDeployLib()
{
    runtime_option_.UseCpu();
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

int YtFastDeployLib::toInitModel(QString ModeDir, int type)
{

    QString model_file = ModeDir + "/inference.pdmodel";
    QString params_file = ModeDir + "/inference.pdiparams";
    QString config_file = ModeDir + "/inference.yml";

    GetLabelList(ModeDir);
    switch (type)
    {
    case kPaddleDetect:
    {
        SetModelType(kPaddleDetect);
        detecte_model_ = new fastdeploy::vision::detection::PaddleDetectionModel(model_file.toStdString().c_str(),\
                                                                                 params_file.toStdString().c_str(),\
                                                                                 config_file.toStdString().c_str(),\
                                                                                 runtime_option_);
        if (!detecte_model_->Initialized())
        {

            qDebug() << "Failed to initialize PaddleDetectionModel." ;
            return kIllegalType;

        }
        return kPaddleDetect;
    }

    case kPaddleSeg:
    {
        SetModelType(kPaddleSeg);
        segment_model_ = new fastdeploy::vision::segmentation::PaddleSegModel(model_file.toStdString().c_str(),\
                                                                              params_file.toStdString().c_str(),\
                                                                              config_file.toStdString().c_str(),\
                                                                              runtime_option_);
        if (!segment_model_->Initialized())
        {

            qDebug() << "Failed to initialize PaddleDetectionModel.";
            return kIllegalType;

        }

        return kPaddleSeg;
    }
    case kPaddleCls:
    {
        SetModelType(kPaddleCls);
        class_model_ = new fastdeploy::vision::classification::PaddleClasModel(model_file.toStdString().c_str(),\
                                                                               params_file.toStdString().c_str(),\
                                                                               config_file.toStdString().c_str(),\
                                                                               runtime_option_);
        if (!class_model_->Initialized())
        {

            qDebug() << "Failed to initialize PaddleDetectionModel.";
            return kIllegalType;

        }
        qDebug() << "success initialized PaddleDetectionModel.";
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
        qDebug()<<"model not init.";
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
        qDebug()<<"start predict detect_model.";
        det_result_.Clear();
        if (detecte_model_->Predict(input_image_, &det_result_))
        {
            qDebug()<<"predict detect_model over.";
            return kPaddleDetect;
        }
        else
        {
            std::cerr << "Failed to predict." << std::endl;
            return kIllegalType;
        }
    }

    if(model_type_ == kPaddleSeg)
    {
        seg_result_.Clear();
        qDebug()<<"start predict seg_model.";
        if (segment_model_->Predict(input_image_, &seg_result_))
        {
            qDebug()<<"predict seg_model over.";
            return kPaddleSeg;
        }
        else
        {
            std::cerr << "Failed to predict." << std::endl;
            return kIllegalType;
        }
    }

    if(model_type_ == kPaddleCls)
    {
        cls_result_.Clear();
        qDebug()<<"start predict cls_model.";

        if (class_model_->Predict(input_image_, &cls_result_))
        {
            qDebug()<<"predict cls_model over.";
            return kPaddleCls;
        }
        else
        {
            qDebug()<< "Failed to predict." ;
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
    for(int i=0; i<det_result_.boxes.size();i++)
    {
        if(det_result_.scores[i] >= getPrsocre)
        {
            InferResult tem_infer_result;
            tem_infer_result.class_id = det_result_.label_ids[i];
            tem_infer_result.confidence = det_result_.scores[i];
            tem_infer_result.lefttop_x = det_result_.boxes[i].at(0);
            tem_infer_result.lefttop_y = det_result_.boxes[i].at(1);
            tem_infer_result.width    = det_result_.boxes[i].at(2) - det_result_.boxes[i].at(0);
            tem_infer_result.height   = det_result_.boxes[i].at(3)- det_result_.boxes[i].at(1);
            if(label_list_.size()>0)
            {
                tem_infer_result.class_name = label_list_[tem_infer_result.class_id];
            }
            GetInferResult.append(tem_infer_result);
        }
    }
}

void YtFastDeployLib::toGetClsData(InferResult &GetInferResult)
{
    if(model_type_ != kPaddleCls)
    {
        qDebug()<<"can't run toGetClsData() with other model type.";
        return;
    }

    auto max_score = std::max_element(cls_result_.scores.begin(), cls_result_.scores.end());
    int max_index;
    if (max_score != cls_result_.scores.end())
    {
        max_index = std::distance(cls_result_.scores.begin(), max_score);
    }

    GetInferResult.confidence = *max_score;
    GetInferResult.class_id = int(cls_result_.label_ids[max_index]);
    if(label_list_.size()>0)
    {
        GetInferResult.class_name = label_list_[GetInferResult.class_id];
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
    all_model_type<<"Detection"<<"Segmentation"<<"Classification";
    if(model_type_ != kIllegalType)
    {
        return all_model_type[model_type_];
    }
    else
    {
        return "No-Init-Model";
    }
}

void YtFastDeployLib::SetModelType(int ModelType)
{
    model_type_ = ModelType;

    if(ModelType == kPaddleDetect)
    {
        class_model_ = nullptr;
        segment_model_ = nullptr;
    }
    else if(ModelType == kPaddleSeg)
    {

        class_model_ = nullptr;
        detecte_model_ = nullptr;

    }
    else if(ModelType == kPaddleCls)
    {
        segment_model_ = nullptr;
        detecte_model_ = nullptr;
    }
}

void YtFastDeployLib::GetLabelList(QString Label_Dir)
{
    if(model_type_ = kIllegalType)
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
            QString Line = label_stream.readLine();
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
