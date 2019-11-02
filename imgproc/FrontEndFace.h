//
// Created by hhg on 19-9-27.
//
#pragma once

#ifndef DVDEFDET_FRONTENDFACE_H
#define DVDEFDET_FRONTENDFACE_H

#include "DV6218Algorithm.h"

class FrontEndFace : public DV6218Algorithm,public DVHoughCircle
{
public:

    ~FrontEndFace();
    FrontEndFace(const FrontEndFace&)=delete;
    FrontEndFace& operator=(const FrontEndFace&)=delete;
    static FrontEndFace& get_instance(){
        static FrontEndFace instance;
        return instance;
    }
private:
    FrontEndFace();

public:
    ErrCode calibFrontEndFace(FrontEndFaceCalibrateParam& fecp,FrontEndFaceCalibrateResult& fecr);
    ErrCode detFrontEndFace(FrontEndFaceDetParam& fedp,FrontEndFaceDetResult& fedr);

    ErrCode calibFrontEndFace2(FrontEndFaceCalibrateParam2& fecp,FrontEndFaceCalibrateResult2& fecr);
    ErrCode detFrontEndFace2(FrontEndFaceDetParam2& fedp,FrontEndFaceDetResult2& fedr);

public:
    //保存校准结果到xml中
    ErrCode saveFrontEndfaceCalibResult(FrontEndFaceCalibrateResult fecr,string filename,string xmlRootName);
    ErrCode saveFrontEndfaceCalibResult2(FrontEndFaceCalibrateParam2 fecp,FrontEndFaceCalibrateResult2 fecr,string filename,string xmlRootName);
    //将xml文件中的参数读入
    ErrCode parseFrontEndfaceCalibResult(string xmlname,FrontEndFaceDetParam& fedp);
    ErrCode parseFrontEndfaceCalibResult2(string xmlname,FrontEndFaceDetParam2& fedp);

private:
    //得到检测的端面圆环的边界线
    void getCirqueUpAndDown(cv::Mat& Hdist,int& rowUpMin,int& rowUpMax,int& rowDownMin,int& rowDownMax,int binLightThreshold);
    //得到端面最外圆的边界上的点集合
    void getEndFaceOuterRadiusPoints(cv::Mat& imgbin,std::vector<Point>& points);
    //得到端面最里面圆的边界上的点集合 水平方向画线扫描
    void getEndFaceInnerRadiusPoints_horizontal(cv::Mat& imgbin,std::vector<Point>& points);
    //得到端面内圆的边界上的点的集合, 垂直方向扫描
    void getEndFaceInnerRadiusPoints_vertical(cv::Mat& imgbin,std::vector<Point>& points);
    //检测亮斑及暗斑缺陷
    void detBrightAndDarkSpot(cv::Mat& imgcartRoi,int kernelType,int stdfilterThreshold,cCParam endFaceDetCC,
            float avgThreshold,FrontEndFaceDetResult& efdr,cv::Mat& imgdist);

    //采用二值化的方式将亮缺陷表现为白色
    void detBrightSpot(cv::Mat& imgsrc,int threshold,cCParam defDetCC,vector<Rect>& defRects);
    //采用二值化的方式 将暗缺陷表现为白色
    void detDarkSpot(cv::Mat& imgsrc,int threshold,cCParam defDetCC,vector<Rect>& defRects);
    //将点转换到原图中
    cv::Point ptChangeToSrcImg(FrontEndFaceDetParam& efdp,cv::Point pointRoi);
    //得到缺陷的最小包围矩形 也包含了将点转换回imgpreset坐标系下
    void getDefBoundingRect(cv::Mat& imgStdfilt,FrontEndFaceDetParam efdp,float imgcartOFFSET,FrontEndFaceDetResult& efdr);
    //数字的区域提取,目前没有加入检测  这里还顺带包括了标记区域的提取
    void detDigital(cv::Mat& imgStdfilt,FrontEndFaceDetParam efdp,FrontEndFaceDetResult& efdr);
    //打印的标记的模板匹配,该模板匹配在原图中抠出的图进行匹配准确率较高,而一般情况下匹配的准确度不高
    void labelTemplateMatching(cv::Mat& imgUpRoi,cv::Mat& imgtemplate,FrontEndFaceDetResult& efdr);
    //得到极坐标展开的图的铆钉铆合的区域
    void getCartRivetArea(cv::Mat& imgRivetArea,FrontEndFaceDetParam efdp,FrontEndFaceDetResult& efdr);
    //检测铆钉铆合情况
    void detRivetArea(FrontEndFaceDetParam efdp,FrontEndFaceDetResult& efdr);
    //将极坐标展开的图像转会imgpreset图的坐标中
    void pointscart2polar(vector<Point>& ploarPoints,float imgcartOFFSET,FrontEndFaceDetResult& efdr);
    //得到铆钉的区域
    void getPresetRivetArea(FrontEndFaceDetParam efdp,FrontEndFaceDetResult& efdr);
    //根据连通域的两个上下限选出特定的数字字母  及打印的标记字符
    void getCharactorsByCC(cv::Mat& img,int detThreshold,cCParam digCC,cCParam digBigCC,vector<Rect>& digitalRects,vector<Mat>& digitalImgs);

    // add new methods
    void calibCharacters(FrontEndFaceCalibrateParam2& fecp, FrontEndFaceCalibrateResult2& fecr);
    void segmentImage(cv::Mat& img_preset, cv::Mat& img_cart_bin, vector<vector<Point>>& contours,
                          cv::Point& cirque_center, float& min_radius, float& max_radius,
                          float& min_theta, float& max_theta, int area_threshold);
    void getBoundingBoxes(vector<vector<Point>>& contours, vector<Rect>& boxes);
    void getMinandMax(int& value, int& minimum, int& maximum);
    void getCharacterParameters(vector<Rect>& boxes, int& min_width, int& max_width,
                                    int& min_height, int& max_height, int& min_gap, int& max_gap);
    void rhoTheta2xy(cv::Point ptCart, cv::Point& ptPolar, cv::Point center, float radius, float min_theta);

    bool detCharacters(FrontEndFaceDetParam2& fedp, FrontEndFaceDetResult2& fedr, int& ith_image);
    void caseJudgement(vector<vector<Point>>& contours, vector<Rect>& boxes, int& upper_bound, int& distance_to_border, int& case_n);


private:
    ErrCode m_err;
    int areaThreshold=3000;
    cCParam fecp_CC;

    // new parameters added
    bool find_starting_point = false;
    int ith_image = 0;
    int counter = 1;
};


#endif //DVDEFDET_FRONTENDFACE_H
