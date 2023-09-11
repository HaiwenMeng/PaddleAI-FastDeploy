#include "ytfastdeploylib.h"
#include <QCoreApplication>
#include <QDebug>
#include  <QImage>



void drawBoundingBox(cv::Mat& image, const cv::Rect& bbox, const QString& category, float confidence)
{
    // 绘制矩形框
    cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 0);

    // 绘制类别和置信度
    std::string label = category.toStdString() + ": " + std::to_string(confidence);
    cv::putText(image, label, cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
}

void CpuInfer(const std::string& model_dir, const std::string& image_file) {
    auto model_file = model_dir + "/inference.pdmodel";
    auto params_file = model_dir + "/inference.pdiparams";
    auto config_file = model_dir + "/inference.yml";

    qDebug()<<"xxxxxxxxxxx line11";
    auto option = fastdeploy::RuntimeOption();
    option.UseCpu();
    //    auto model = fastdeploy::vision::segmentation::PaddleSegModel(model_file, params_file,
    //                                                        config_file, option);

//    auto model = fastdeploy::vision::detection::PaddleDetectionModel(model_file, params_file,
//                                                                     config_file, option);

    auto model = fastdeploy::vision::classification::PaddleClasModel(model_file, params_file,
                                                                     config_file, option);

    if (!model.Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return;
    }
    auto im = cv::imread(image_file);


    fastdeploy::vision::ClassifyResult res;
    if (!model.Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
    }
    qDebug()<<"xxxxxxxxxxx line30";
    std::cout << res.Str() << std::endl;
    auto vis_im = fastdeploy::vision::VisClassification(im, res);
//    cv::imwrite("SegOcr_result.jpg", vis_im);
    cv::imshow("SegOcr_result",vis_im);
    cv::waitKey(0);
    std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}


void FastdeployInferDet(std::string& in_model_dir, std::string& in_img_dir)

{
    YtFastDeployLib MyFastDeployObj;
    QVector<InferResult> my_infer_result;
    QString model_dir = QString::fromStdString(in_model_dir);
    QString img_dir = QString::fromStdString(in_img_dir);
    if(0 == MyFastDeployObj.toInitModel(model_dir, 0))
    {
        qDebug()<<MyFastDeployObj.toGetModelType();

        QImage input_image = QImage(img_dir);
        if(0==MyFastDeployObj.toPredect(input_image))
        {
            MyFastDeployObj.toGetDetcData(my_infer_result, 0.5);
        }

        qDebug()<<QString(u8"id:%1, 置信度:%2").arg(my_infer_result[0].class_id ).arg(my_infer_result[0].confidence);
        cv::Rect show_react(my_infer_result[0].lefttop_x,my_infer_result[0].lefttop_y,\
                my_infer_result[0].width, my_infer_result[0].height);

        QStringList labels = QStringList();
        labels<<"speedlimit"<<"crosswalk"<<"trafficlight"<<"stop";
        QString tem_label = labels.at(my_infer_result[0].class_id);

        cv::Mat mat;
        MyFastDeployObj.Qimage2cvMat(&input_image, mat);

        drawBoundingBox(mat, show_react, tem_label,my_infer_result[0].confidence);
        cv::imshow("show_pictrue", mat);
        cv::waitKey(0);


    }

}

void FastdeployInferCls(std::string& in_model_dir, std::string& in_img_dir)

{
    YtFastDeployLib MyFastDeployObj;
    InferResult my_infer_result;
    QString model_dir = QString::fromStdString(in_model_dir);
    QString img_dir = QString::fromStdString(in_img_dir);
    if(2 == MyFastDeployObj.toInitModel(model_dir, 2))
    {
        qDebug()<<MyFastDeployObj.toGetModelType();

        QImage input_image = QImage(img_dir);

        //预处理,仅用来显示，不对输入图像做转换
        input_image = input_image.convertToFormat(QImage::Format_RGB888 );
        cv::Mat mat(input_image.height(), input_image.width(), CV_8UC3, (void*)input_image.constBits(), input_image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
        cv::imshow("converted from QImage", mat);
        cv::waitKey(0);



        //toPredect()内已实集成了QIMage转cv::Mat的操作
        if(2==MyFastDeployObj.toPredect(input_image))
        {
            MyFastDeployObj.toGetClsData(my_infer_result);
        }

        qDebug()<<QString(u8"id:%1, 置信度:%2").arg(my_infer_result.class_id ).arg(my_infer_result.confidence);

        //结果显示
        cv::putText(mat, std::to_string(my_infer_result.class_id)+" : "+std::to_string(my_infer_result.confidence),\
                    cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        cv::imshow("converted from QImage", mat);
        cv::waitKey(0);
    }
}

void FastdeployInferSeg(std::string& in_model_dir, std::string& in_img_dir)

{
    YtFastDeployLib MyFastDeployObj;
    InferResult my_infer_result;
    QString model_dir = QString::fromStdString(in_model_dir);
    QString img_dir = QString::fromStdString(in_img_dir);
    if(1 == MyFastDeployObj.toInitModel(model_dir, 1))
    {
        qDebug()<<"mhw_log1111";
        qDebug()<<MyFastDeployObj.toGetModelType();

        QImage input_image = QImage(img_dir);

        //预处理,仅用来显示，不对输入图像做转换
        cv::Mat mat;
        MyFastDeployObj.Qimage2cvMat(&input_image, mat);
        cv::imshow("converted from QImage", mat);
        cv::waitKey(0);


        QImage seg_mask;
        if(1==MyFastDeployObj.toPredect(input_image))
        {
            MyFastDeployObj.toGetSegData(seg_mask);
        }

        //结果显示
        cv::Mat cv_mask;
        MyFastDeployObj.Qimage2cvMat(&seg_mask, cv_mask);
        cv::imwrite("mask.png",cv_mask);
        cv::imshow("mask", cv_mask);
        cv::waitKey(0);

    }
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    //detect
//    std::string det_model_dir("F:/paddlex_beta/history_project/det_ppyoloe_crn_l_20230829/output/best_model/infer");
//    std::string det_img_dir("F:/paddlex_beta/history_project/det_ppyoloe_crn_l_20230829/output/best_model/infer/road101.png");
//    FastdeployInferDet(det_model_dir, det_img_dir);

    //seg
    std::string seg_model_dir("F:/PaddleX210336_workdir/741934/3/output/infer_model");
    std::string seg_img_dir("F:/paddlex_beta/WorkDir/741934/2/data/example_data/seg_optic_examples/JPEGImages/H0002.jpg");
    FastdeployInferSeg(seg_model_dir, seg_img_dir);

    //cls
//    std::string cls_model_dir("F:/paddlex_beta/history_project/cls_PPLCNet_x1_0_20230901/output/best_model/infer");
//    std::string cls_img_path("F:/paddlex_beta/datasets/example_data/cls_flowers_examples/images/image_06576.jpg");
//    FastdeployInferCls(cls_model_dir, cls_img_path);

//    CpuInfer(cls_model_dir, cls_img_path);

//    qDebug()<<"MHW_LOG444444";




    return a.exec();
}
