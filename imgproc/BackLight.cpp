//
// Created by hhg on 19-9-27.
//

#include "BackLight.h"


//! calibration
ErrCode BackLight::calibBackLight(BackLightCalibrateParam& blcp, BackLightCalibrateResult& blcr)
{
    cv::Mat imgGrey;
    if(blcp.imgSrc.channels()==3){
        cvtColor(blcp.imgSrc,imgGrey,CV_BGR2GRAY);
    } else {
        imgGrey = blcp.imgSrc;
    }
    cv::Mat imgPreset = imgGrey(blcp.preSetArea);       //!< if necessary, crop the image using a rectangle box. otherwise using the original image
    cv::Point innerCirqueCenter;                        //!< the center for the inner cirque
    float innerRadius  = 700;                           //!< the radius of the inner cirque

    if (hasSample(imgGrey, blcp.avgGrayThreshold))
    {
        getCenterAndRadius(imgPreset, blcp.binThreshold, blcp.cCAreaThreshold, innerCirqueCenter, innerRadius);
        blcr.innerRadius = innerRadius;
        blcr.innerCenter = innerCirqueCenter;
        getRangeOfRegion(blcp, blcr, innerRadius);
        blcr.rOuterRadius = innerRadius * (blcp.rMax / blcp.rMin);
        return Err_NoErr;
    }
    return Err_CalibBackLight;
}

//! detection
ErrCode BackLight::detBackLight(BackLightDetParam& bldp, BackLightDetResult& bldr)
{
    cv::Mat imgGrey;
    if(bldp.imgSrc.channels()==3){
        cvtColor(bldp.imgSrc, imgGrey, CV_BGR2GRAY);
    } else{
        imgGrey = bldp.imgSrc;
    }
    cv::Mat imgPreset = imgGrey(bldp.blcp.preSetArea);              //!< if necessary, crop the image using a rectangle box. otherwise using the original image
    cv::Point innerCirqueCenter;                                    //!< the center for the inner cirque
    float innerRadius  = 0.0;                                       //!< the radius of the inner cirque

    //! check if there is a sample
    if (hasSample(imgGrey, bldp.blcp.avgGrayThreshold))
    {
        getCenterAndRadius(imgPreset, bldp.blcp.binThreshold,
                           bldp.blcp.cCAreaThreshold,
                           innerCirqueCenter, innerRadius);
        if(getDistance(innerCirqueCenter, bldp.blcr.innerCenter) > 100)//TODO:如果两个圆的圆心之间的距离差的太远,则报错
        {
            dzlog_error("getDistance(innerCirqueCenter, bldp.blcr.innerCenter) > 100 ...");
            return Err_DetBackLight;
         };

         //! calculate region
        getRangeOfRegion(bldp.blcp, bldp.blcr, innerRadius);
        float rOuterRadius;
        rOuterRadius=innerRadius*(bldp.blcp.rMax / bldp.blcp.rMin);

        //! segment region
        cv::Mat imgCart;
        polar2cart(imgPreset, imgCart,
                   innerCirqueCenter, rOuterRadius);

        //! the whole region
        cv::Mat imgRoi;
        getRoiImg(imgCart, bldp.blcr.rRoiLow,
                  bldp.blcr.rRoiUp, imgRoi);

        //! rivet region
        cv::Mat imgRivet;
        getRoiImg(imgCart, bldp.blcr.rRivetLow,
                  bldp.blcr.rRivetUp, imgRivet);

        //! lower steel ball region
        cv::Mat imgInnerDetRoi;
        getRoiImg(imgCart, bldp.blcr.rInnerDetRoiLow,
                  bldp.blcr.rInnerDetRoiUp, imgInnerDetRoi);

        //! outer steel ball region
        cv::Mat imgOuterDetRoi;
        getRoiImg(imgCart, bldp.blcr.rOuterDetRoiLow,
                  bldp.blcr.rOuterDetRoiUp, imgOuterDetRoi);

        if(bldp.imgSrc.channels()==1)
        {
            cvtColor(bldp.imgSrc, bldr.imgResult, CV_GRAY2BGR);
        } else{
            bldr.imgResult = bldp.imgSrc;
        }

        //! rivet defect detection
        vector<Point> imgRivetDefPointsOnImgSrc;
        getRivetDefPointsOnImgSrc(imgRivet, bldp, innerCirqueCenter,
                                  rOuterRadius, imgRivetDefPointsOnImgSrc);
        if(imgRivetDefPointsOnImgSrc.size()>100)
        {
            bldr.hasDef = true;
            bldr.deft   = LossRivet;
        }

        //! inner steel ball region
        vector<Point> innerDetRoiDefPointsOnImgSrc;
        getLossStellDefPointsOnImgSrc(imgInnerDetRoi, bldp, innerCirqueCenter,
                                      rOuterRadius, innerDetRoiDefPointsOnImgSrc);

        //! outer steel ball region
        vector<Point> outerDetRoiDefPointsOnImgSrc;
        getLossStellDefPointsOnImgSrc(imgOuterDetRoi, bldp, innerCirqueCenter,
                                      rOuterRadius, outerDetRoiDefPointsOnImgSrc);

        if(innerDetRoiDefPointsOnImgSrc.size() > 100 || outerDetRoiDefPointsOnImgSrc.size() > 100)
        {
            bldr.hasDef = true;
            bldr.deft   = LossStellBall;
        }

        drawDefPointsOnImgResult(bldr.imgResult, imgRivetDefPointsOnImgSrc);
        drawDefPointsOnImgResult(bldr.imgResult, innerDetRoiDefPointsOnImgSrc);
        drawDefPointsOnImgResult(bldr.imgResult, outerDetRoiDefPointsOnImgSrc);

        return Err_NoErr;
    }
    return Err_NoErr;
}


//保存校准结果到xml中
ErrCode BackLight::saveBackLightCalibResult(BackLightCalibrateParam& blcp,BackLightCalibrateResult blcr, std::string filename, std::string xmlRootName)
{
    createXML(filename.c_str(),xmlRootName.c_str());
//    createXML("efcr.xml","CalibResult");
    XMLDocument doc;
    int res=doc.LoadFile(filename.c_str());
    if(res!=0)
    {
        dzlog_error("load xml file failed");
        return Err_SaveXml;
    }

    XMLElement *root=doc.RootElement();

    XMLElement *blcrnode=doc.NewElement("BackLightCalibrateResult");
    root->InsertEndChild(blcrnode);

    XMLElement* preSetArea=doc.NewElement("preSetArea");
    blcrnode->InsertEndChild(preSetArea);
    XMLElement* preSetArea_x=doc.NewElement("X");
    XMLElement* preSetArea_y=doc.NewElement("Y");
    XMLElement* preSetArea_width=doc.NewElement("Width");
    XMLElement* preSetArea_height=doc.NewElement("Height");
    preSetArea->InsertEndChild(preSetArea_x);
    preSetArea->InsertEndChild(preSetArea_y);
    preSetArea->InsertEndChild(preSetArea_width);
    preSetArea->InsertEndChild(preSetArea_height);
    XMLText* preSetArea_texX=doc.NewText(to_string(blcp.preSetArea.x).c_str());
    XMLText* preSetArea_texY=doc.NewText(to_string(blcp.preSetArea.y).c_str());
    XMLText* preSetArea_texWidth=doc.NewText(to_string(blcp.preSetArea.width).c_str());
    XMLText* preSetArea_texHeight=doc.NewText(to_string(blcp.preSetArea.height).c_str());
    preSetArea_x->InsertEndChild(preSetArea_texX);
    preSetArea_y->InsertEndChild(preSetArea_texY);
    preSetArea_width->InsertEndChild(preSetArea_texWidth);
    preSetArea_height->InsertEndChild(preSetArea_texHeight);

//    int avgGrayThreshold = 200;         //!< 平均灰度阈值 threshold for average gray level. this is used to check if there is a sample.
//    int binThreshold     = 100;         //!< 二值化阈值      threshold for binarization
//    int cCAreaThreshold  = 1000;        //!< 连通域阈值      threshold for connected components. This is to remove small connected components
//    int roiBinThreshold  = 40;         //!< ROI二值化阈值   threshold for binarization after transform (circular area to rectangle area)
    XMLElement *avgGrayThreshold=doc.NewElement("avgGrayThreshold");
    blcrnode->InsertEndChild(avgGrayThreshold);
    XMLText* tex_avgGrayThreshold=doc.NewText(to_string(blcp.avgGrayThreshold).c_str());
    avgGrayThreshold->InsertEndChild(tex_avgGrayThreshold);

    XMLElement *binThreshold=doc.NewElement("binThreshold");
    blcrnode->InsertEndChild(binThreshold);
    XMLText* tex_binThreshold=doc.NewText(to_string(blcp.binThreshold).c_str());
    binThreshold->InsertEndChild(tex_binThreshold);

    XMLElement *cCAreaThreshold=doc.NewElement("cCAreaThreshold");
    blcrnode->InsertEndChild(cCAreaThreshold);
    XMLText* tex_cCAreaThreshold=doc.NewText(to_string(blcp.cCAreaThreshold).c_str());
    cCAreaThreshold->InsertEndChild(tex_cCAreaThreshold);

    XMLElement *roiBinThreshold=doc.NewElement("roiBinThreshold");
    blcrnode->InsertEndChild(roiBinThreshold);
    XMLText* tex_roiBinThreshold=doc.NewText(to_string(blcp.roiBinThreshold).c_str());
    roiBinThreshold->InsertEndChild(tex_roiBinThreshold);

//    float rMin=45.0;                      //6218轴承的内半径
//    float rMax=80.0;                     //6218轴承的外半径
//    float innerCirqueHeight=11.0;         //轴承的内圆环的宽度
//    float rivetCirqueHeight=13.0;         //轴承的内 外圆环中间的铆合区域的宽度
//    float outerCirqueHeight=11.0;         //轴承的外圆环的宽度
//    float rRivetLowOffsetRatio=0.05;      //铆钉检测区域相对于铆合区域的内侧偏移比率
//    float rRivetUpOffsetRatio=0.11;        //铆钉检测区域相对于铆合区域的外侧偏移比率
    XMLElement *rMin=doc.NewElement("rMin");
    blcrnode->InsertEndChild(rMin);
    XMLText* tex_rMin=doc.NewText(to_string(blcp.rMin).c_str());
    rMin->InsertEndChild(tex_rMin);

    XMLElement *rMax=doc.NewElement("rMax");
    blcrnode->InsertEndChild(rMax);
    XMLText* tex_rMax=doc.NewText(to_string(blcp.rMax).c_str());
    rMax->InsertEndChild(tex_rMax);

    XMLElement *innerCirqueHeight=doc.NewElement("innerCirqueHeight");
    blcrnode->InsertEndChild(innerCirqueHeight);
    XMLText* tex_innerCirqueHeight=doc.NewText(to_string(blcp.innerCirqueHeight).c_str());
    innerCirqueHeight->InsertEndChild(tex_innerCirqueHeight);

    XMLElement *rivetCirqueHeight=doc.NewElement("rivetCirqueHeight");
    blcrnode->InsertEndChild(rivetCirqueHeight);
    XMLText* tex_rivetCirqueHeight=doc.NewText(to_string(blcp.rivetCirqueHeight).c_str());
    rivetCirqueHeight->InsertEndChild(tex_rivetCirqueHeight);

    XMLElement *outerCirqueHeight=doc.NewElement("outerCirqueHeight");
    blcrnode->InsertEndChild(outerCirqueHeight);
    XMLText* tex_outerCirqueHeight=doc.NewText(to_string(blcp.outerCirqueHeight).c_str());
    outerCirqueHeight->InsertEndChild(tex_outerCirqueHeight);

    XMLElement *rRivetLowOffsetRatio=doc.NewElement("rRivetLowOffsetRatio");
    blcrnode->InsertEndChild(rRivetLowOffsetRatio);
    XMLText* tex_rRivetLowOffsetRatio=doc.NewText(to_string(blcp.rRivetLowOffsetRatio).c_str());
    rRivetLowOffsetRatio->InsertEndChild(tex_rRivetLowOffsetRatio);

    XMLElement *rRivetUpOffsetRatio=doc.NewElement("rRivetUpOffsetRatio");
    blcrnode->InsertEndChild(rRivetUpOffsetRatio);
    XMLText* tex_rRivetUpOffsetRatio=doc.NewText(to_string(blcp.rRivetUpOffsetRatio).c_str());
    rRivetUpOffsetRatio->InsertEndChild(tex_rRivetUpOffsetRatio);




    XMLElement *tOffset=doc.NewElement("tOffset");
    blcrnode->InsertEndChild(tOffset);
    XMLText* tex_tOffset=doc.NewText(to_string(blcp.tOffset).c_str());
    tOffset->InsertEndChild(tex_tOffset);

//    cCParam clear_noise_cc;
    XMLElement* clear_noise_cc=doc.NewElement("clear_noise_cc");
    blcrnode->InsertEndChild(clear_noise_cc);
    XMLElement* clear_noise_cc_STAT_AREA=doc.NewElement("STAT_AREA");
    XMLElement* clear_noise_cc_STAT_WIDTH=doc.NewElement("STAT_WIDTH");
    XMLElement* clear_noise_cc_STAT_HEIGHT=doc.NewElement("STAT_HEIGHT");
    clear_noise_cc->InsertEndChild(clear_noise_cc_STAT_AREA);
    clear_noise_cc->InsertEndChild(clear_noise_cc_STAT_WIDTH);
    clear_noise_cc->InsertEndChild(clear_noise_cc_STAT_HEIGHT);
    XMLText* tex_clear_noise_cc_STAT_AREA=doc.NewText(to_string(blcp.clear_noise_cc.STAT_AREA).c_str());
    XMLText* tex_clear_noise_cc_STAT_WIDTH=doc.NewText(to_string(blcp.clear_noise_cc.STAT_WIDTH).c_str());
    XMLText* tex_clear_noise_cc_STAT_HEIGHT=doc.NewText(to_string(blcp.clear_noise_cc.STAT_HEIGHT).c_str());

    clear_noise_cc_STAT_AREA->InsertEndChild(tex_clear_noise_cc_STAT_AREA);
    clear_noise_cc_STAT_WIDTH->InsertEndChild(tex_clear_noise_cc_STAT_WIDTH);
    clear_noise_cc_STAT_HEIGHT->InsertEndChild(tex_clear_noise_cc_STAT_HEIGHT);


//    float innerRadius;                  //!< 内圆环内径      the inner radius of the inner cirque
//    Point innerCenter=cv::Point(0,0);                  //内圆的圆心
//    int rMax;                               //最外圆的半径
//    int rRoiLow;                           //!< ROI下界      the lower boundary for the cirque and rivet area (the whole ROI in rectangle area)
//    int rRoiUp;                           //!< ROI上界      the upper boundary for the cirque and rivet area (the whole ROI in rectangle area)
//    int rRivetLow;                      //!< 铆钉区域下界 the lower boundary for the rivet area (rivet area extraction in rectangle area)
//    int rRivetUp;                      //!< 铆钉区域上界 the upper boundary for the rivet area (rivet area ectraction in rectangle area)
//    int rInnerDetRoiLow;                //检测内圈透光情况的范围上边界
//    int rInnerDetRoiUp;                 //检测内圈透光情况的范围下边界
//    int rOuterDetRoiLow;                //检测外圈透光情况的范围上边界
//    int rOuterDetRoiUp;                 //检测外圈透光情况的范围下边界

    XMLElement *innerRadius=doc.NewElement("innerRadius");
    blcrnode->InsertEndChild(innerRadius);
    XMLText* tex_innerRadius=doc.NewText(to_string(blcr.innerRadius).c_str());
    innerRadius->InsertEndChild(tex_innerRadius);

    XMLElement *innerCenter=doc.NewElement("innerCenter");
    blcrnode->InsertEndChild(innerCenter);
    XMLElement* innerCenter_x=doc.NewElement("X");
    XMLElement* innerCenter_y=doc.NewElement("Y");
    innerCenter->InsertEndChild(innerCenter_x);
    innerCenter->InsertEndChild(innerCenter_y);
    XMLText* tex_innerCenter_x=doc.NewText(to_string(blcr.innerCenter.x).c_str());
    XMLText* tex_innerCenter_y=doc.NewText(to_string(blcr.innerCenter.y).c_str());
    innerCenter_x->InsertEndChild(tex_innerCenter_x);
    innerCenter_y->InsertEndChild(tex_innerCenter_y);


//    int rMax;                               //最外圆的半径
//    int rRoiLow;                           //!< ROI下界      the lower boundary for the cirque and rivet area (the whole ROI in rectangle area)
//    int rRoiUp;                           //!< ROI上界      the upper boundary for the cirque and rivet area (the whole ROI in rectangle area)
//    int rRivetLow;                      //!< 铆钉区域下界 the lower boundary for the rivet area (rivet area extraction in rectangle area)
//    int rRivetUp;                      //!< 铆钉区域上界 the upper boundary for the rivet area (rivet area ectraction in rectangle area)
//    int rInnerDetRoiLow;                //检测内圈透光情况的范围上边界
//    int rInnerDetRoiUp;                 //检测内圈透光情况的范围下边界
//    int rOuterDetRoiLow;                //检测外圈透光情况的范围上边界
//    int rOuterDetRoiUp;                 //检测外圈透光情况的范围下边界


    XMLElement *rOuterRadius=doc.NewElement("rOuterRadius");
    blcrnode->InsertEndChild(rOuterRadius);
    XMLText* tex_rOuterRadius=doc.NewText(to_string(blcr.rOuterRadius).c_str());
    rOuterRadius->InsertEndChild(tex_rOuterRadius);

    XMLElement *rRoiLow=doc.NewElement("rRoiLow");
    blcrnode->InsertEndChild(rRoiLow);
    XMLText* tex_rRoiLow=doc.NewText(to_string(blcr.rRoiLow).c_str());
    rRoiLow->InsertEndChild(tex_rRoiLow);

    XMLElement *rRoiUp=doc.NewElement("rRoiUp");
    blcrnode->InsertEndChild(rRoiUp);
    XMLText* tex_rRoiUp=doc.NewText(to_string(blcr.rRoiUp).c_str());
    rRoiUp->InsertEndChild(tex_rRoiUp);

    XMLElement *rRivetLow=doc.NewElement("rRivetLow");
    blcrnode->InsertEndChild(rRivetLow);
    XMLText* tex_rRivetLow=doc.NewText(to_string(blcr.rRivetLow).c_str());
    rRivetLow->InsertEndChild(tex_rRivetLow);

    XMLElement *rRivetUp=doc.NewElement("rRivetUp");
    blcrnode->InsertEndChild(rRivetUp);
    XMLText* tex_rRivetUp=doc.NewText(to_string(blcr.rRivetUp).c_str());
    rRivetUp->InsertEndChild(tex_rRivetUp);

    XMLElement *rInnerDetRoiLow=doc.NewElement("rInnerDetRoiLow");
    blcrnode->InsertEndChild(rInnerDetRoiLow);
    XMLText* tex_rInnerDetRoiLow=doc.NewText(to_string(blcr.rInnerDetRoiLow).c_str());
    rInnerDetRoiLow->InsertEndChild(tex_rInnerDetRoiLow);

    XMLElement *rInnerDetRoiUp=doc.NewElement("rInnerDetRoiUp");
    blcrnode->InsertEndChild(rInnerDetRoiUp);
    XMLText* tex_rInnerDetRoiUp=doc.NewText(to_string(blcr.rInnerDetRoiUp).c_str());
    rInnerDetRoiUp->InsertEndChild(tex_rInnerDetRoiUp);

    XMLElement *rOuterDetRoiLow=doc.NewElement("rOuterDetRoiLow");
    blcrnode->InsertEndChild(rOuterDetRoiLow);
    XMLText* tex_rOuterDetRoiLow=doc.NewText(to_string(blcr.rOuterDetRoiLow).c_str());
    rOuterDetRoiLow->InsertEndChild(tex_rOuterDetRoiLow);

    XMLElement *rOuterDetRoiUp=doc.NewElement("rOuterDetRoiUp");
    blcrnode->InsertEndChild(rOuterDetRoiUp);
    XMLText* tex_rOuterDetRoiUp=doc.NewText(to_string(blcr.rOuterDetRoiUp).c_str());
    rOuterDetRoiUp->InsertEndChild(tex_rOuterDetRoiUp);

    doc.SaveFile(filename.c_str());
    return Err_NoErr;
}

//将xml文件中的参数读入
ErrCode BackLight::parseBackLightCalibResult(std::string xmlname, BackLightDetParam& bldp)
{
    XMLDocument doc;
    int res=doc.LoadFile(xmlname.c_str());
    if(res!=0)
    {
        dzlog_error("load xml file failed");
        return Err_ParseXml;
    }

    XMLElement *root=doc.RootElement();
    XMLElement* occrnode=root->FirstChildElement("BackLightCalibrateResult");

    //filletRoiCC
    XMLElement* preSetArea=occrnode->FirstChildElement("preSetArea");
    XMLElement* preSetArea_X=preSetArea->FirstChildElement("X");
    XMLElement* preSetArea_Y=preSetArea->FirstChildElement("Y");
    XMLElement* preSetArea_Width=preSetArea->FirstChildElement("Width");
    XMLElement* preSetArea_Height=preSetArea->FirstChildElement("Height");
    bldp.blcp.preSetArea.x=atoi(preSetArea_X->GetText());
    bldp.blcp.preSetArea.y=atoi(preSetArea_Y->GetText());
    bldp.blcp.preSetArea.width=atoi(preSetArea_Width->GetText());
    bldp.blcp.preSetArea.height=atoi(preSetArea_Height->GetText());

    XMLElement* avgGrayThreshold=occrnode->FirstChildElement("avgGrayThreshold");
    bldp.blcp.avgGrayThreshold=atoi(avgGrayThreshold->GetText());

    XMLElement* binThreshold=occrnode->FirstChildElement("binThreshold");
    bldp.blcp.binThreshold=atoi(binThreshold->GetText());

    XMLElement* cCAreaThreshold=occrnode->FirstChildElement("cCAreaThreshold");
    bldp.blcp.cCAreaThreshold=atoi(cCAreaThreshold->GetText());

    XMLElement* roiBinThreshold=occrnode->FirstChildElement("roiBinThreshold");
    bldp.blcp.roiBinThreshold=atoi(roiBinThreshold->GetText());

    XMLElement* rMin=occrnode->FirstChildElement("rMin");
    bldp.blcp.rMin=atoi(rMin->GetText());

    XMLElement* rMax=occrnode->FirstChildElement("rMax");
    bldp.blcp.rMax=atoi(rMax->GetText());

    XMLElement* innerCirqueHeight=occrnode->FirstChildElement("innerCirqueHeight");
    bldp.blcp.innerCirqueHeight=atoi(innerCirqueHeight->GetText());

    XMLElement* rivetCirqueHeight=occrnode->FirstChildElement("rivetCirqueHeight");
    bldp.blcp.rivetCirqueHeight=atoi(rivetCirqueHeight->GetText());

    XMLElement* outerCirqueHeight=occrnode->FirstChildElement("outerCirqueHeight");
    bldp.blcp.outerCirqueHeight=atoi(outerCirqueHeight->GetText());

    XMLElement* rRivetLowOffsetRatio=occrnode->FirstChildElement("rRivetLowOffsetRatio");
    bldp.blcp.rRivetLowOffsetRatio=atof(rRivetLowOffsetRatio->GetText());

    XMLElement* rRivetUpOffsetRatio=occrnode->FirstChildElement("rRivetUpOffsetRatio");
    bldp.blcp.rRivetUpOffsetRatio=atof(rRivetUpOffsetRatio->GetText());

    XMLElement* tOffset=occrnode->FirstChildElement("tOffset");
    bldp.blcp.tOffset=atoi(tOffset->GetText());


//    clear_noise_cc
    XMLElement* clear_noise_cc=occrnode->FirstChildElement("clear_noise_cc");
    XMLElement* clear_noise_cc_STAT_AREA=clear_noise_cc->FirstChildElement("STAT_AREA");
    XMLElement* clear_noise_cc_STAT_WIDTH=clear_noise_cc->FirstChildElement("STAT_WIDTH");
    XMLElement* clear_noise_cc_STAT_HEIGHT=clear_noise_cc->FirstChildElement("STAT_HEIGHT");
    bldp.blcp.clear_noise_cc.STAT_AREA=atoi(clear_noise_cc_STAT_AREA->GetText());
    bldp.blcp.clear_noise_cc.STAT_WIDTH=atoi(clear_noise_cc_STAT_WIDTH->GetText());
    bldp.blcp.clear_noise_cc.STAT_HEIGHT=atoi(clear_noise_cc_STAT_HEIGHT->GetText());

    XMLElement* innerRadius=occrnode->FirstChildElement("innerRadius");
    bldp.blcr.innerRadius=atoi(innerRadius->GetText());

    XMLElement* innerCenter=occrnode->FirstChildElement("innerCenter");
    XMLElement* innerCenter_x=innerCenter->FirstChildElement("X");
    XMLElement* innerCenter_y=innerCenter->FirstChildElement("Y");
    bldp.blcr.innerCenter.x=atoi(innerCenter_x->GetText());
    bldp.blcr.innerCenter.y=atoi(innerCenter_y->GetText());

    XMLElement* rOuterRadius=occrnode->FirstChildElement("rOuterRadius");
    bldp.blcr.rOuterRadius=atoi(rOuterRadius->GetText());
    XMLElement* rRoiLow=occrnode->FirstChildElement("rRoiLow");
    bldp.blcr.rRoiLow=atoi(rRoiLow->GetText());
    XMLElement* rRoiUp=occrnode->FirstChildElement("rRoiUp");
    bldp.blcr.rRoiUp=atoi(rRoiUp->GetText());
    XMLElement* rRivetLow=occrnode->FirstChildElement("rRivetLow");
    bldp.blcr.rRivetLow=atoi(rRivetLow->GetText());
    XMLElement* rRivetUp=occrnode->FirstChildElement("rRivetUp");
    bldp.blcr.rRivetUp=atoi(rRivetUp->GetText());

    XMLElement* rInnerDetRoiLow=occrnode->FirstChildElement("rInnerDetRoiLow");
    bldp.blcr.rInnerDetRoiLow=atoi(rInnerDetRoiLow->GetText());

    XMLElement* rInnerDetRoiUp=occrnode->FirstChildElement("rInnerDetRoiUp");
    bldp.blcr.rInnerDetRoiUp=atoi(rInnerDetRoiUp->GetText());

    XMLElement* rOuterDetRoiLow=occrnode->FirstChildElement("rOuterDetRoiLow");
    bldp.blcr.rOuterDetRoiLow=atoi(rOuterDetRoiLow->GetText());

    XMLElement* rOuterDetRoiUp=occrnode->FirstChildElement("rOuterDetRoiUp");
    bldp.blcr.rOuterDetRoiUp=atoi(rOuterDetRoiUp->GetText());

    return Err_NoErr;
}


bool BackLight::hasSample(cv::Mat img, int avgThreshold)
{
    float avgGrayLevel = 0.0;               //!< mean of the grayscale image
    float stdGrayLevel = 0.0;               //!< standard deviation of the grayscale image

    GetGrayAvgStdDev(img, avgGrayLevel, stdGrayLevel);
    dzlog_debug("avgGrayLevel = %f",avgGrayLevel);
    dzlog_debug("stdGrayLevel = %f",stdGrayLevel);

    if (avgGrayLevel > avgThreshold)
    {
        cout << "There is no sample. Please Check!" << endl;
        return false;
    }
    return true;
}


void BackLight::getCenterAndRadius(cv::Mat img, int binThreshold, int cCAreaThreshold, cv::Point& center, float& radius)
{
    cv::Mat imgBin;                                     //!< binarized image
    std::vector<cv::Point> innerCirquePoints;           //!< vector container for the edge points of the inner cirque

    //! threshold the image blcp.detThreshold
    threshold(img, imgBin, binThreshold, 255, CV_THRESH_BINARY);
    showImg("imgBin", imgBin);
    Mat element = getStructuringElement(MORPH_RECT,
                                        Size(9, 9));
    Mat open_result;
    morphologyEx(imgBin, open_result, MORPH_OPEN,element);

    cv::Mat close_result;
    morphologyEx(open_result,close_result,MORPH_CLOSE,element);

    //! get the center and the radius of the inner ring
    getBackLightPlaneInnerRadiusPoints(close_result, innerCirquePoints);
    dzlog_debug("innerCirquePoints.size() == %d",innerCirquePoints.size());
    circleLeastFit(innerCirquePoints, center, radius);
    dzlog_debug("innerCirqueCenter.X =%d Y=%d",center.x,center.y);
    dzlog_debug("innerRadius = %f",radius);
}


void BackLight::getBackLightPlaneInnerRadiusPoints(cv::Mat imgbin, std::vector<cv::Point>& points)
{
    int midcols = imgbin.cols / 2;
    int midrows = imgbin.rows / 2;

    for (int i = midrows; i > 0; i--)
    {
        if (imgbin.ptr<uchar>(i)[midcols] > 0)
        {
            for (int j = midcols; j > 0; j--)
            {
                if (imgbin.ptr<uchar>(i)[j] < 20)
                {
                    points.push_back(cv::Point(j, i));
                    break;
                }
            }

            for (int k = midcols; k < imgbin.cols - 1; k++)
            {
                if (imgbin.ptr<uchar>(i)[k] < 20)
                {
                    points.push_back(cv::Point(k, i));
                    break;
                }
            }
        }
        else
        {
            break;
        }
    }
}


void BackLight::transformAndThreshold(cv::Mat img, cv::Mat& dst, cv::Point center, int rMax, int roiBinThreshold)
{
    cv::Mat imgCart;                //!< transformed image.
    cv::Mat imgCartBin;             //!< binarized image after transform

    //! transform the circular area into rectangle area
    polar2cart(img, imgCart, center, rMax);
    showImg("imgCart", imgCart);
    threshold(imgCart, dst, roiBinThreshold, 255, CV_THRESH_BINARY);
    showImg("imgCartBin", dst);
}


void BackLight::getRangeOfRegion(BackLightCalibrateParam& blcp,BackLightCalibrateResult& blcr, float radius)
{
    float rROiLowRatio=(2.0+blcp.innerCirqueHeight/blcp.rMin)/2.0;//(1+1.24)/2 = 1.12
    float rRoiUpRatio=(1+(blcp.innerCirqueHeight+blcp.rivetCirqueHeight)/blcp.rMin + 1 +
            (blcp.innerCirqueHeight+blcp.rivetCirqueHeight+blcp.outerCirqueHeight)/blcp.rMin)/2.0;//(1.53+1.777)/2=1.65
    //铆钉铆合区域提取
    float rRivetLowRatio=(1+blcp.innerCirqueHeight/blcp.rMin)+blcp.rRivetLowOffsetRatio;//1.24+0.05 内侧的透光光斑较窄 = 1.29
    float rRivetUpRatio=(1+(blcp.innerCirqueHeight+blcp.rivetCirqueHeight)/blcp.rMin)-blcp.rRivetUpOffsetRatio;//1.53-0.11 外侧的透光光斑较宽 =1.42

    float rInnerDetRoiLowRatio=(2.0+blcp.innerCirqueHeight/blcp.rMin)/2.0;//(1+1.24)/2=1.12
    float rInnerDetRoiUpRatio=(2+(2*blcp.innerCirqueHeight+blcp.rivetCirqueHeight)/blcp.rMin)/2.0;//(1.24+1.53)/2=1.388

    float rOuterDetRoiLowRatio=(2+(2*blcp.innerCirqueHeight+blcp.rivetCirqueHeight)/blcp.rMin)/2.0;//(1.24+1.53)/2=1.388
    float rOuterDetRoiUpRatio=(2+(2*(blcp.innerCirqueHeight+blcp.rivetCirqueHeight)+blcp.outerCirqueHeight)/blcp.rMin)/2.0;//(1.53+1.177)/2=1.65

    dzlog_debug("rROiLowRatio ==%f",rROiLowRatio);
    dzlog_debug("rRoiUpRatio ==%f",rRoiUpRatio);
    dzlog_debug("rRivetLowRatio ==%f",rRivetLowRatio);
    dzlog_debug("rRivetUpRatio ==%f",rRivetUpRatio);
    dzlog_debug("rInnerDetRoiLowRatio ==%f",rInnerDetRoiLowRatio);
    dzlog_debug("rInnerDetRoiUpRatio ==%f",rInnerDetRoiUpRatio);
    dzlog_debug("rOuterDetRoiLowRatio ==%f",rOuterDetRoiLowRatio);
    dzlog_debug("rOuterDetRoiUpRatio ==%f",rOuterDetRoiUpRatio);

    blcr.rRoiLow = (int)(rROiLowRatio * radius);
    blcr.rRoiUp = (int)(rRoiUpRatio * radius);
    blcr.rRivetLow = (int)(rRivetLowRatio * radius);
    blcr.rRivetUp = (int)(rRivetUpRatio * radius);
    blcr.rInnerDetRoiLow=(int)(rInnerDetRoiLowRatio * radius);
    blcr.rInnerDetRoiUp=(int)(rInnerDetRoiUpRatio * radius);
    blcr.rOuterDetRoiLow=(int)(rOuterDetRoiLowRatio * radius);
    blcr.rOuterDetRoiUp=(int)(rOuterDetRoiUpRatio * radius);

    dzlog_debug("blcr.rRoiLow == %d",blcr.rRoiLow);
    dzlog_debug("blcr.rRoiUp == %d",blcr.rRoiUp);
    dzlog_debug("blcr.rRivetLow == %d",blcr.rRivetLow);
    dzlog_debug("blcr.rRivetUp == %d",blcr.rRivetUp);
    dzlog_debug("blcr.rInnerDetRoiLow == %d",blcr.rInnerDetRoiLow);
    dzlog_debug("blcr.rInnerDetRoiUp == %d",blcr.rInnerDetRoiUp);
    dzlog_debug("blcr.rOuterDetRoiLow == %d",blcr.rOuterDetRoiLow);
    dzlog_debug("blcr.rOuterDetRoiUp == %d",blcr.rOuterDetRoiUp);

}


//void BackLight::detectDefect(cv::Mat& src,cv::Mat& img,BackLightCalibrateParam& blcp, BackLightCalibrateResult& blcr, cv::Point center)
//{
//    int tOffset;
//    cv::Mat shiftedImg;
//    cv::Mat imgRoi = img(cv::Rect(0, blcr.rRoiLow, img.cols, blcr.rMax - blcr.rRoiLow));
//    cv::Mat imgRoi2 = imgRoi.clone();
//    showImg("imgRoi1", imgRoi);
//
//    //! convert grayscale image into RGB image
//    cvtColor(src, src, COLOR_GRAY2BGR);
//
//    //! normal Detection
//    detect(src, imgRoi, blcp, blcr, center, 0);
//
//    //! shift the image and detect. tOffset can be calculate by x = pi * rMax * blcp.tOffset / 180
//    tOffset = blcp.tOffset * CV_PI * blcr.rMax / 180;
//
//    shiftImage(imgRoi2, shiftedImg, tOffset);
//    showImg("shiftedImg", shiftedImg);
//    detect(src, shiftedImg, blcp, blcr, center, tOffset);
//}


bool BackLight::isDefect(cv::Mat& imgsrc, cCParam& ccp_lb, cCParam& ccp_ub)
{
    int count = 0;
    cv::Mat labels, img_color, stats, centroids;
    int nccomps = cv::connectedComponentsWithStats(imgsrc, labels, stats, centroids);

    for (int num = 1; num < nccomps; num++)
    {
        int STAT_AREA   = stats.at<int>(num, CC_STAT_AREA);
        int STAT_WIDTH  = stats.at<int>(num, CC_STAT_WIDTH);
        int STAT_HEIGHT = stats.at<int>(num, CC_STAT_HEIGHT);
        int STAT_LEFT   = stats.at<int>(num, CC_STAT_LEFT);
        int STAT_TOP    = stats.at<int>(num, CC_STAT_TOP);

        cv::Rect temp(STAT_LEFT, STAT_TOP, STAT_WIDTH, STAT_HEIGHT);

        cout << "num = " << num << endl;
        cout << "area = " << STAT_AREA << endl;
        cout << "width = " << STAT_WIDTH << endl;
        cout << "height = " << STAT_HEIGHT << endl;

        if ((STAT_WIDTH > ccp_lb.STAT_WIDTH) && (STAT_WIDTH < ccp_ub.STAT_WIDTH) && (STAT_AREA > ccp_lb.STAT_AREA) && (STAT_AREA < ccp_ub.STAT_AREA))
        {
            count++;
        }
        else
        {
            imgsrc(temp) = 0;
        }
    }

    if (count > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


void BackLight::shiftImage(cv::Mat& img, cv::Mat& dst, int LOffset)
{
    dst = 0 * img.clone();

    for (int i = 0; i < img.cols; i++)
    {
        for (int j = 0; j < img.rows; j++)
        {
            if ((i + LOffset) < (img.cols))
            {
                dst.ptr<uchar>(j)[i] = img.ptr<uchar>(j)[i + LOffset];
            }
            else
            {
                dst.ptr<uchar>(j)[i] = img.ptr<uchar>(j)[i - img.cols + LOffset];
            }
        }
    }
}


void BackLight::findGradient(cv::Mat& img, cv::Mat& dst)
{
    cv::Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(img, dst, MORPH_GRADIENT, element, Point(-1, -1), 2);
    showImg("Grad", dst);
}


//void BackLight::detect(cv::Mat& src, cv::Mat& img, BackLightCalibrateParam& blcp,BackLightCalibrateResult& blcr, cv::Point center, int tOffset)
//{
//    cv::Mat imgRivet;               //!< the image of the rivet
//    cv::Mat imgRivetGrad;           //!< the morphological gradient of the rivet
//
//    bool detectSteelBall = 0;
//    bool detectRivet = 0;
//
//    //! extract rivet area
//    imgRivet = img(cv::Rect(0, blcr.rRivetMin - blcr.rMin, img.cols, blcr.rRivetMax - blcr.rRivetMin)).clone();
//    showImg("imgRivet", imgRivet);
//
//    //! remove small connected components
//    bwareaopen(imgRivet, blcp.cCAreaThreshold);
//    showImg("imgRoi3", imgRivet);
//
//    //! check if rivet is lost
//    detectRivet = isDefect(imgRivet, blcp.rivet_lb_cc, blcp.rivet_ub_cc);
//    cout << "detectRivet = " << detectRivet << endl;
//
//    cv::Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
//    dilate(img, img, element, Point(-1, -1), 1);
//    bwareaopen(img, blcp.cCAreaThreshold);
//    showImg("imgRoi4", img);
//
//    //! check if steel ball is lost
//    detectSteelBall = isDefect(img, blcp.steelBall_lb_cc, blcp.steelBall_ub_cc);
//    showImg("imgRoi5", img);
//
//    if (detectRivet)
//    {
//        if (detectSteelBall)
//        {
//            img(cv::Rect(0, blcr.rRivetMin, img.cols, blcr.rRivetMax - blcr.rRivetMin)) = imgRivet;
//            findGradient(img, img);
//            plotDefectArea(src, img, CV_LOCAL_AREA, blcr.rMin, tOffset, center, blcr.rMax);
//            showImg("src", src);
//        }
//        else
//        {
//            findGradient(imgRivet, imgRivet);
//            plotDefectArea(src, imgRivet, CV_LOCAL_AREA, blcr.rRivetMin, tOffset, center, blcr.rMax);
//            showImg("src", src);
//        }
//    }
//    else
//    {
//        if (detectSteelBall)
//        {
//            findGradient(img, img);
//            showImg("Gra", img);
//            plotDefectArea(src, img, CV_LOCAL_AREA, blcr.rMin, tOffset, center, blcr.rMax);
//            showImg("src", src);
//        }
//    }
//}




//void BackLight::showCircle(std::string name, cv::Mat img, cv::Point center, float radius)
//{
//    if (img.channels() < 3)
//        cvtColor(img, img, COLOR_GRAY2BGR);
//
//    circle(img, center, radius, Scalar(0, 255, 0), 2);
//    showImg("name", img);
//}

void BackLight::getRoiImg(cv::Mat imgGrey,int up,int down,cv::Mat& imgdist)
{
    imgdist=imgGrey(cv::Rect(0,up,imgGrey.cols,down-up));
}

void BackLight::detLossRivet(cv::Mat& imgRivet,cCParam rivetLowCC,cCParam rivetUpCC,vector<Rect>& defRects)
{
    cv::Mat imgBin;
    threshold(imgRivet,imgBin,200,255,CV_THRESH_BINARY);
//    showImg("imgRivetBin",imgBin);
    vector<Rect> tempDefRects;
    bwareaopen(imgBin,rivetLowCC,tempDefRects);
    for(int i=0;i<tempDefRects.size();i++)
    {
        if(tempDefRects[i].width<rivetUpCC.STAT_WIDTH && tempDefRects[i].height<rivetUpCC.STAT_HEIGHT)
        {
            defRects.push_back(tempDefRects[i]);
        }
    }
}

void BackLight::detLossStellBall(cv::Mat& imgSrc,cCParam stellBallCC,vector<Rect>& defRects)
{
    cv::Mat imgBin;
    threshold(imgSrc,imgBin,50,255,CV_THRESH_BINARY);
//    showImg("imgLossStellBin",imgBin);
    bwareaopen(imgBin,stellBallCC,defRects);
    showImg("imgLossStellBin_bwopen",imgBin);
}

void BackLight::getLossStellDefPointsOnImgSrc(cv::Mat imgSrc,BackLightDetParam bldp,cv::Point center,int rOuterRadius,vector<Point>& defPointsOnImgSrc)
{
    vector<Rect> defRects;
    detLossStellBall(imgSrc, bldp.steelBall_lb_cc, defRects);
    vector<Point> defPointsOnDetImg;
    vector<Point> defPointsOnHoleCart;
    if(defRects.size()>0)
    {
        getRectsBorderPoints(defRects, defPointsOnDetImg);
        points2HoleCartImg(defPointsOnDetImg, bldp.blcr.rInnerDetRoiUp,
                           defPointsOnHoleCart);
        pointsHoleCart2ImgSrc(defPointsOnHoleCart, center,
                              rOuterRadius, bldp.blcp.preSetArea,
                              defPointsOnImgSrc);
    }else{
        cv::Mat imgInnerDetRoiLowShiftDist;
        int LOffset=bldp.blcp.tOffset/360.0*imgSrc.cols;
        shiftImage(imgSrc, imgInnerDetRoiLowShiftDist, LOffset);
        detLossStellBall(imgInnerDetRoiLowShiftDist, bldp.steelBall_lb_cc, defRects);
        if(defRects.size()>0)
        {
            getRectsBorderPoints(defRects, defPointsOnDetImg);
            points2HoleCartImg(defPointsOnDetImg, bldp.blcr.rInnerDetRoiUp,
                               bldp.blcp.tOffset, imgSrc.cols,defPointsOnHoleCart);
            pointsHoleCart2ImgSrc(defPointsOnHoleCart, center,
                                  rOuterRadius, bldp.blcp.preSetArea,
                                  defPointsOnImgSrc);
        }
    }
}

void BackLight::getRivetDefPointsOnImgSrc(cv::Mat imgSrc, BackLightDetParam bldp,
                                          cv::Point center, int rOuterRadius,
                                          vector<Point>& defPointsOnImgSrc)
{
    vector<Rect> imgRivetRects;
    detLossRivet(imgSrc,bldp.rivet_lb_cc, bldp.rivet_ub_cc, imgRivetRects);
    vector<Point> rivetDefPointsOnDetImg;
    vector<Point> rivetDefPointsOnHoleCart;

    if(imgRivetRects.size()>0)
    {
        getRectsBorderPoints(imgRivetRects, rivetDefPointsOnDetImg);
        points2HoleCartImg(rivetDefPointsOnDetImg, bldp.blcr.rRivetUp,
                           rivetDefPointsOnHoleCart);
        pointsHoleCart2ImgSrc(rivetDefPointsOnHoleCart, center,
                              rOuterRadius, bldp.blcp.preSetArea,
                              defPointsOnImgSrc);
    }else{
        cv::Mat imgRivetShiftDist;
        int LOffset=bldp.blcp.tOffset/360.0*imgSrc.cols;
        shiftImage(imgSrc, imgRivetShiftDist, LOffset);
        detLossRivet(imgRivetShiftDist, bldp.rivet_lb_cc,
                     bldp.rivet_ub_cc, imgRivetRects);
        if(imgRivetRects.size()>0)
        {
            getRectsBorderPoints(imgRivetRects, rivetDefPointsOnDetImg);
            points2HoleCartImg(rivetDefPointsOnDetImg, bldp.blcr.rRivetUp,
                               rivetDefPointsOnHoleCart);
            pointsHoleCart2ImgSrc(rivetDefPointsOnHoleCart, center,
                                  rOuterRadius, bldp.blcp.preSetArea,
                                  defPointsOnImgSrc);
        }
    }
}

