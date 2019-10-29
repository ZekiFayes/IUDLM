//
// Created by hhg on 19-9-27.
//
#pragma once

//! includes
#include "DV6218Algorithm.h"


//! class
class BackLight : public DV6218Algorithm
{
public:
    ~BackLight() {};
    BackLight(const BackLight&) = delete;
    BackLight& operator=(const BackLight&) = delete;
    static BackLight& get_instance()
    {
        static BackLight instance;
        return instance;
    }

private:
    BackLight() {};

public:

    //主校准函数
    ErrCode calibBackLight(BackLightCalibrateParam& blcp, BackLightCalibrateResult& bclr);
    //主检测函数
    ErrCode detBackLight(BackLightDetParam& bldp, BackLightDetResult& bldr);
    //保存校准结果到xml中
    ErrCode saveBackLightCalibResult(BackLightCalibrateParam& blcp,BackLightCalibrateResult blcr, std::string filename, std::string xmlRootName);
    //将xml文件中的参数读入
    ErrCode parseBackLightCalibResult(std::string xmlname, BackLightDetParam& bldp);

private:
    bool hasSample(cv::Mat img, int avgtThreshold);
    void getCenterAndRadius(cv::Mat img, int binThreshold, int cCAreaThreshold, cv::Point& center, float& radius);
    void getBackLightPlaneInnerRadiusPoints(cv::Mat imgbin, std::vector<cv::Point>& points);
    void getRangeOfRegion(BackLightCalibrateParam& blcp,BackLightCalibrateResult& bclr, float radius);
    void transformAndThreshold(cv::Mat img, cv::Mat& dst, cv::Point center, int rMax, int roiBinThreshold);
    void shiftImage(cv::Mat& img, cv::Mat& dst, int LOffset);//LOffset 水平方向的长度 移动距离
//    void detectDefect(cv::Mat& src, cv::Mat& img,BackLightCalibrateParam& blcp, BackLightCalibrateResult& blcr, cv::Point center);
    void detect(cv::Mat& src, cv::Mat& img, BackLightCalibrateParam& blcp, BackLightCalibrateResult& blcr,cv::Point center, int tOffset);
    bool isDefect(cv::Mat& imgsrc, cCParam& ccp_lb, cCParam& ccp_ub);
    void findGradient(cv::Mat& img, cv::Mat& dst);
    void getRoiImg(cv::Mat imgGrey,int up,int down,cv::Mat& imgdist);
    void detLossRivet(cv::Mat& imgRivet,cCParam rivetLowCC,cCParam rivetUpCC,vector<Rect>& defRects);
    void detLossStellBall(cv::Mat& imgSrc,cCParam stellBallCC,vector<Rect>& defRects);
    void showCircle(std::string name, cv::Mat img, cv::Point center, float radius);

    void getLossStellDefPointsOnImgSrc(cv::Mat imgSrc,BackLightDetParam bldp,cv::Point center,int rOuterRadius,vector<Point>& defPointsOnImgSrc);

    void getRivetDefPointsOnImgSrc(cv::Mat imgSrc,BackLightDetParam bldp,cv::Point center,int rOuterRadius,vector<Point>& defPointsOnImgSrc);


private:
    ErrCode m_err;
};


