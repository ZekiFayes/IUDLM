//
// Created by hhg on 19-9-27.
//

#include "FrontEndFace.h"


FrontEndFace::FrontEndFace()
{
    fecp_CC.STAT_AREA=1000;         //连通域大小 0.1*imgsrc.rows*imgsrc.cols
    fecp_CC.STAT_WIDTH=600;         //连通域宽度 比imgsrc.cols 稍微小一点
    fecp_CC.STAT_HEIGHT=30;         //连通域高度 比50高一点
}

FrontEndFace::~FrontEndFace()
{

}

/**
 * 端面的校准主要目的是得到端面四条分界线的半径
 * @param fecp
 * @param fecr
 */
ErrCode FrontEndFace::calibFrontEndFace(FrontEndFaceCalibrateParam& fecp,FrontEndFaceCalibrateResult& fecr)
{
    showImg("imgsrc",fecp.imgsrc);
    cv::Mat imgPreset=fecp.imgsrc(fecp.rectPreset);
    showImg("imgPreset",imgPreset);
    float avgGrayLevel=0.0;//初始化
    float stdGreyLevel=0.0;//初始化
    //原图的平均灰度为55
    GetGrayAvgStdDev(fecp.imgsrc,avgGrayLevel,stdGreyLevel);
    dzlog_debug("avgGrayLevel == %f",avgGrayLevel);
//    dzlog_debug("stdGreyLevel == %f",stdGreyLevel);
    if(avgGrayLevel < fecp.avgLightThreshold)//TODO:这里要查看实际工况 如果检测到的全图平均阈值小于fecp.avgLightThreshold 则报错,没有样件
    {
        dzlog_error("图像的平均灰度小于亮度阈值,没有校准件");
        m_err=Err_CalibFrontEndFace;
    }
    cv::Mat imgbin;

    threshold(imgPreset,imgbin,fecp.detThreshold,255,CV_THRESH_BINARY_INV);//将preset区域二值化
    showImg("imgbin",imgbin);
//    cv::Mat imgbitwisenot;
//    bitwise_not(imgbin,imgbitwisenot);
//    showImg("imgbitwisenot",imgbitwisenot);
    bwareaopen(imgbin,fecp.fecp_CC);//TODO:这里是去除较小的连通域的操作,目的是只检测大圆环区域
    showImg("imgbin_afterbw",imgbin);

//    cv::Mat outerbin=imgbin.clone();
    vector<Point> outerCirquePoints;
//    imwrite("endfacebin.jpg",imgbin);
    getEndFaceOuterRadiusPoints(imgbin,outerCirquePoints);
    dzlog_debug("outerCirquePoints.size() == %d",outerCirquePoints.size());

    //拟合圆 求取圆心及半径
    float outerRadius=0;//这里是在检测之前先赋一次值,防止没有检测的情况;这里的情况没有用
    cv::Point outerCirqueCenter=cv::Point(0,0);
    if(!circleLeastFit(outerCirquePoints,outerCirqueCenter,outerRadius))
    {
        dzlog_error("最小二乘法拟合圆错误 ...");
        return Err_CircleLeastFit;
    }
    dzlog_debug("outerCirqueCenter.x==%d,y== %d",outerCirqueCenter.x,outerCirqueCenter.y);
    dzlog_debug("outerRadius == %f",outerRadius);

    fecr.outerRmaxThreshold=outerRadius+fecp.rOFFSETThreshold;
    fecr.outerRminThreshold=outerRadius-fecp.rOFFSETThreshold;
//    sort(outerCirquePoints.begin(),outerCirquePoints.end(),points_x_sort);
//    cv::Point pt1=outerCirquePoints[0];
//    cv::Point pt2=outerCirquePoints[outerCirquePoints.size()-1];
//    float angleLeft=atan2(outerCirqueCenter.y-pt1.y,pt1.x-outerCirqueCenter.x);
//    float angleRight=atan2(outerCirqueCenter.y-pt2.y,pt2.x-outerCirqueCenter.x);
//    dzlog_debug("angleleft == %f",angleLeft);
//    dzlog_debug("angleRight == %f",angleRight);

    cv::Mat imgcart;
//    float thetaMin=CV_PI/3.0;
//    float thetaMax=2.0*CV_PI/3.0;
    float rmin=0.5*outerRadius;//TODO:这里要指定最小半径的地方,极坐标转换的时候不需要将整个图都转换过来,只需要目标感兴趣区域转换过来即可

    //FIXME:这里为什么要加0.1
    polar2cart(imgPreset, imgcart, outerCirqueCenter, outerRadius,fecp.angleRight,fecp.angleLeft,rmin);
    showImg("imgcart",imgcart);

    vector<Point> innerCirquePoints;
    innerCirquePoints.clear();
//    showImg("imgbin2",imgbin);
//    getEndFaceInnerRadiusPoints_horizontal(imgbin,innerCirquePoints);
    getEndFaceInnerRadiusPoints_vertical(imgbin,innerCirquePoints);
    dzlog_debug("innerCirquePoints.size() == %d",innerCirquePoints.size());
    if(innerCirquePoints.size() < 100)
    {
        dzlog_error("detInnerCirque Error ...");
    }
    float innerRadius=500;//初始化系数,这里是没用的
    cv::Point innerCirqueCenter=cv::Point(0,0);
    if(!circleLeastFit(innerCirquePoints,innerCirqueCenter,innerRadius))
    {
        dzlog_error("最小二乘法拟合圆错误 ...");
        return Err_CircleLeastFit;
    }
    sort(innerCirquePoints.begin(),innerCirquePoints.end(),points_x_sort);
//    cv::Point pt3=innerCirquePoints[0];
//    cv::Point pt4=innerCirquePoints[innerCirquePoints.size()-1];
    dzlog_debug("innerCirqueCenter.x == %d,y == %d",innerCirqueCenter.x,innerCirqueCenter.y);
    dzlog_debug("innerRadius == %f",innerRadius);
    fecr.innerRmaxThreshold=innerRadius+fecp.rOFFSETThreshold;
    fecr.innerRminThreshold=innerRadius-fecp.rOFFSETThreshold;
    dzlog_debug("fecr.innerRminThreshold == %d",fecr.innerRminThreshold);
    dzlog_debug("fecr.innerRmaxThreshold == %d",fecr.innerRmaxThreshold);
    fecr.outerCirque_outerRadius=outerRadius;
    fecr.innerCirque_innerRadius=innerRadius;
    fecr.innerRadiusCenter=innerCirqueCenter;
    fecr.outerRadiusCenter=outerCirqueCenter;
    fecr.centerDist=DVAlgorithm::getDistance(innerCirqueCenter,outerCirqueCenter) +20;
    dzlog_debug("fecr.centerDist == %d",fecr.centerDist);
    cv::Mat imgcartBin;
    threshold(imgcart,imgcartBin,fecp.detThreshold,255,CV_THRESH_BINARY_INV);//二值化提取目标检测区域
    bwareaopen(imgcartBin,fecp.fecp_CC);//去除小的连通域
    showImg("imgcartBin",imgcartBin);
    cv::Mat imgcartbin_bitwise;
    bitwise_not(imgcartBin,imgcartbin_bitwise);
    showImg("imgcartbin_bitwise",imgcartbin_bitwise);
    bwareaopen(imgcartbin_bitwise,fecp.fecp_CC);
    showImg("imgcartbin_bitwise2",imgcartbin_bitwise);
//    imwrite("imgcartbin_bitwise.jpg",imgcartbin_bitwise);
    int rowUpMin_max,rowUpMax_min,rowDownMin_max,rowDownMax_min;
    int rowUpMin_min,rowUpMax_max,rowDownMin_min,rowDownMax_max;
    cv::Mat Hdist = Mat::zeros(imgcartbin_bitwise.rows, 1, CV_32FC1);//CV_32FC1
    reduce(imgcartbin_bitwise, Hdist, 1, CV_REDUCE_SUM, CV_32FC1);//水平方向投影
    Hdist = Hdist / imgcart.cols;
    getCirqueUpAndDown(Hdist,rowUpMin_max,rowUpMax_min,rowDownMin_max,rowDownMax_min,220);
    getCirqueUpAndDown(Hdist,rowUpMin_min,rowUpMax_max,rowDownMin_min,rowDownMax_max,20);
#if 0
    cv::Mat Hdist = Mat::zeros(imgcartBin.rows, 1, CV_32FC1);//CV_32FC1
    reduce(imgcartBin, Hdist, 1, CV_REDUCE_SUM, CV_32FC1);//水平方向投影
    Hdist = Hdist / imgcart.cols;
//    std::cout << "Hdist == " << Hdist << endl;
    int rowUpMin=Hdist.rows/2;//最内侧的环形区域
    int rowUpMax=0;
    int rowUpMin2=Hdist.rows/2;//最外侧的环形区域
    int rowUpMax2=0;
    int rowDownMin=Hdist.rows;
    int rowDownMax=Hdist.rows/2;
    int rowDownMin2=Hdist.rows;
    int rowDownMax2=Hdist.rows/2;
    for (int i=Hdist.rows/2;i>0;i--)
    {
        if (Hdist.at<float>(i, 0) < 50)
        {
            if(rowUpMin>i){
                rowUpMin=i;
            }
            if(rowUpMax<i)
            {
                rowUpMax=i;
            }
        }
    }

    for(int i = Hdist.rows/2; i < Hdist.rows; i++)
    {
        if(Hdist.at<float>(i, 0) < 50)
        {
            if(rowDownMin>i)
            {
                rowDownMin=i;
            }
            if(rowDownMax<i)
            {
                rowDownMax=i;
            }
        }
    }

    for (int i=Hdist.rows/2;i>0;i--)
    {
        if (Hdist.at<float>(i, 0) < 150)
        {
            if(rowUpMin2>i){
                rowUpMin2=i;
            }
            if(rowUpMax2<i)
            {
                rowUpMax2=i;
            }
        }
    }

    for(int i = Hdist.rows/2; i < Hdist.rows; i++)
    {
        if(Hdist.at<float>(i, 0) < 150)
        {
            if(rowDownMin2>i)
            {
                rowDownMin2=i;
            }
            if(rowDownMax2<i)
            {
                rowDownMax2=i;
            }
        }
    }
#endif
    dzlog_debug("rowUpMin_max == %d",rowUpMin_max);
    dzlog_debug("rowUpMax_min == %d",rowUpMax_min);
    dzlog_debug("rowDownMin_max == %d",rowDownMin_max);
    dzlog_debug("rowDownMax_min == %d",rowDownMax_min);

    dzlog_debug("rowUpMin_min == %d",rowUpMin_min);
    dzlog_debug("rowUpMax_max == %d",rowUpMax_max);
    dzlog_debug("rowDownMin_min == %d",rowDownMin_min);
    dzlog_debug("rowDownMax_max == %d",rowDownMax_max);
    //如果选出的边界差值大于10  说明
    if(rowUpMin_max-rowUpMin_min>fecp.cirqueLineThreshold || rowUpMax_max-rowUpMax_min>fecp.cirqueLineThreshold
       ||rowDownMin_max-rowDownMin_min>fecp.cirqueLineThreshold || rowDownMax_max-rowDownMax_min>fecp.cirqueLineThreshold)
    {
        dzlog_error("环状边界倾斜,原始外圆边界有缺陷,剔除 ...");
    }

    fecr.outerCirqueHeight=rowDownMax_min-rowDownMin_max;//外环的高度
    fecr.innerCirqueHeight=rowUpMax_min-rowUpMin_max;//内环的高度
    dzlog_debug("fecr.outerCirqueHeight == %d",fecr.outerCirqueHeight);
    dzlog_debug("fecr.innerCirqueHeight == %d",fecr.innerCirqueHeight);
    fecr.outerCirqueHeightMin=rowDownMax_min-rowDownMin_max-fecp.heightOFFSETThreshold;//外圆最外线的最小半径范围
    fecr.outerCirqueHeightMax=rowDownMax_min-rowDownMin_max+ fecp.heightOFFSETThreshold;//外圆最外线的最大半径范围
    fecr.innerCirqueHeightMin=rowUpMax_min-rowUpMin_max - fecp.heightOFFSETThreshold;//内圆最内线的最小半径范围
    fecr.innerCirqueHeightMax=rowUpMax_min-rowUpMin_max + fecp.heightOFFSETThreshold;//内圆最内径的最大半径范围
    dzlog_debug("fecr.outerCirqueHeightMin == %d",fecr.outerCirqueHeightMin);//答应阈值半径阈值范围
    dzlog_debug("fecr.outerCirqueHeightMax == %d",fecr.outerCirqueHeightMax);
    dzlog_debug("fecr.innerCirqueHeightMin == %d",fecr.innerCirqueHeightMin);
    dzlog_debug("fecr.innerCirqueHeightMax == %d",fecr.innerCirqueHeightMax);
    cv::Mat imgRoiUp=imgcart(cv::Rect(0,rowUpMin_max,imgcart.cols,rowUpMax_min-rowUpMin_max));
    cv::Mat imgRoiDown=imgcart(cv::Rect(0,rowDownMin_max,imgcart.cols,rowDownMax_min-rowDownMin_max));
    showImg("imgRoiDown",imgRoiDown);
    showImg("imgRoiUp",imgRoiUp);

#if 0 //检测ROI cirque区域的平均灰度
    int imgRoiUp_sumThreshold=0;
    for(int i=rowUpMin_max;i<rowUpMax_min;i++)
    {
//        dzlog_debug("Hdist.at<float>(%d, 0) == %f",i,Hdist.at<float>(i, 0));
        imgRoiUp_sumThreshold+=Hdist.at<float>(i, 0);
    }
    float imgRoiUp_avgTh=imgRoiUp_sumThreshold/(rowUpMax_min-rowUpMin_max);
    dzlog_debug("imgRoiUp_avgTh == %f",imgRoiUp_avgTh);//上圆环检测区域的平均灰度阈值

    int imgRoiDown_sumThreshold=0;
    for(int i=rowDownMin_max;i<rowDownMax_min;i++)
    {
        imgRoiDown_sumThreshold+=Hdist.at<float>(i, 0);
    }
    float imgRoiDown_avgTh=imgRoiDown_sumThreshold/(rowDownMax_min-rowDownMin_max);
    dzlog_debug("imgRoiDown_avgTh == %f",imgRoiDown_avgTh);
    fecr.innerCirqueGreyThreshold=imgRoiUp_avgTh;
    fecr.outerCirqueGreyThreshold=imgRoiDown_avgTh;
    fecr.detThreshold=(imgRoiUp_avgTh+imgRoiDown_avgTh)/2 - 50 > 60?(imgRoiUp_avgTh+imgRoiDown_avgTh)/2 - 50:60;//阈值最小为60  这里是在实际平均灰度的基础上-50 为了保证所有点都可以检出
    dzlog_debug("fecr.detThreshold == %d",fecr.detThreshold);
#endif
    //计算平均灰度,这个方法应该消耗的资源会比较多,速度不够快
    cv::Mat mean;
    cv::Mat stdDev;
    cv::meanStdDev(imgRoiDown, mean, stdDev);
    dzlog_debug("imgRoiDown_avgThreshold== %f",mean.ptr<double>(0)[0]);
    float imgRoiDownAvgThreshold=mean.ptr<double>(0)[0];
    cv::Mat mean1;
    cv::Mat stdDev1;
    cv::meanStdDev(imgRoiUp, mean1, stdDev1);
    dzlog_debug("imgRoiUp_avgThreshold == %f",mean1.ptr<double>(0)[0]);
    float imgRoiUpAvgThreshold=mean1.ptr<double>(0)[0];
    fecr.innerCirqueGreyThreshold=imgRoiUpAvgThreshold;
    fecr.outerCirqueGreyThreshold=imgRoiDownAvgThreshold;
    fecr.detThreshold=((imgRoiUpAvgThreshold+imgRoiDownAvgThreshold)/2 - 90) > 60?((imgRoiUpAvgThreshold+imgRoiDownAvgThreshold)/2 - 90):60;//阈值最小为60  这里是在实际平均灰度的基础上-90 为了保证所有点都可以检出
    dzlog_debug("fecr.detThreshold == %d",fecr.detThreshold);
    fecr.outerCirque_innerRadius=outerRadius-(rowDownMax_min-rowDownMin_max);
    fecr.innerCirque_outerRadius=innerRadius+ (rowUpMax_min-rowUpMin_max);
    return Err_NoErr;
}

ErrCode FrontEndFace::calibFrontEndFace2(FrontEndFaceCalibrateParam2& fecp,FrontEndFaceCalibrateResult2& fecr)
{
//    showImg("imgsrc",fecp.imgSrc);
    if(!fecp.imgSrc.data)
    {
        dzlog_error("!fecp.imgSrc.data");
    }
    cv::Mat imgPreset=fecp.imgSrc(fecp.preSetArea);
//    showImg("imgPreset",imgPreset);
    cv::Mat imgbin;
    threshold(imgPreset,imgbin,fecp.componentExistThreshold,255,CV_THRESH_BINARY_INV);//将preset区域二值化
//    showImg("imgbin",imgbin);
    bwareaopen(imgbin,fecp_CC);//fixme:这里是去除较小的连通域的操作,目的是只检测大圆环区域
//    showImg("imgbin_afterbw",imgbin);
    vector<Point> outerCirquePoints;
    getEndFaceOuterRadiusPoints(imgbin,outerCirquePoints);
    dzlog_debug("outerCirquePoints.size() == %d",outerCirquePoints.size());
    //拟合圆 求取圆心及半径
    float outerRadius=0;//这里是在检测之前先赋一次值,防止没有检测的情况;这里的情况没有用
    cv::Point outerCirqueCenter=cv::Point(0,0);
    if(!circleLeastFit(outerCirquePoints,outerCirqueCenter,outerRadius))
    {
        dzlog_error("最小二乘法拟合圆错误 ... ");
        return Err_CircleLeastFit;
    }
    dzlog_debug("outerCirqueCenter.x==%d,y== %d",outerCirqueCenter.x,outerCirqueCenter.y);
    dzlog_debug("outerRadius == %f",outerRadius);

    vector<Point> innerCirquePoints;
    innerCirquePoints.clear();
    getEndFaceInnerRadiusPoints_vertical(imgbin,innerCirquePoints);
    dzlog_debug("innerCirquePoints.size() == %d",innerCirquePoints.size());
    if(innerCirquePoints.size() < 100)
    {
        dzlog_error("detInnerCirque Error ...");
    }
    float innerRadius=0.0;//初始化系数,这里是没用的
    cv::Point innerCirqueCenter=cv::Point(0,0);
    if(!circleLeastFit(innerCirquePoints,innerCirqueCenter,innerRadius))
    {
        dzlog_error("拟合内圆错误 ...");
        return Err_CircleLeastFit;
    }
    sort(innerCirquePoints.begin(),innerCirquePoints.end(),points_x_sort);
//    cv::Point pt3=innerCirquePoints[0];
//    cv::Point pt4=innerCirquePoints[innerCirquePoints.size()-1];
    dzlog_debug("innerCirqueCenter.x == %d,y == %d",innerCirqueCenter.x,innerCirqueCenter.y);
    dzlog_debug("innerRadius == %f",innerRadius);

    cv::Mat imgcart;
//    float thetaMin=CV_PI/3.0;
//    float thetaMax=2.0*CV_PI/3.0;
    float rmin=0.5*outerRadius;//TODO:这里要指定最小半径的地方,极坐标转换的时候不需要将整个图都转换过来,只需要目标感兴趣区域转换过来即可

    //FIXME:这里为什么要加0.1
    polar2cart(imgPreset, imgcart, outerCirqueCenter, outerRadius,fecp.angleRight,fecp.angleLeft,rmin);
    showImg("imgcart",imgcart);


    fecr.outerCirque_outerRadius=outerRadius;
    fecr.innerCirque_innerRadius=innerRadius;
    fecr.innerRadiusCenter=innerCirqueCenter;
    fecr.outerRadiusCenter=outerCirqueCenter;


    cv::Mat imgcartBin;
    threshold(imgcart,imgcartBin,fecp.componentExistThreshold,255,CV_THRESH_BINARY_INV);//二值化提取目标检测区域
//    showImg("imgcartBin",imgcartBin);
    bwareaopen(imgcartBin,fecp_CC);//去除端面小的连通域的干扰
//    showImg("imgcartBin",imgcartBin);
    cv::Mat imgcartbin_bitwise;
    bitwise_not(imgcartBin,imgcartbin_bitwise);
//    showImg("imgcartbin_bitwise",imgcartbin_bitwise);
    bwareaopen(imgcartbin_bitwise,fecp_CC);
//    showImg("imgcartbin_bitwise2",imgcartbin_bitwise);
    int rowUpMin_max,rowUpMax_min,rowDownMin_max,rowDownMax_min;
//    int rowUpMin_min,rowUpMax_max,rowDownMin_min,rowDownMax_max;
    cv::Mat Hdist = Mat::zeros(imgcartBin.rows, 1, CV_32FC1);//CV_32FC1
    reduce(imgcartbin_bitwise, Hdist, 1, CV_REDUCE_SUM, CV_32FC1);//水平方向投影
    Hdist = Hdist / imgcart.cols;
    cout<<Hdist<<endl;
    getCirqueUpAndDown(Hdist,rowUpMin_max,rowUpMax_min,rowDownMin_max,rowDownMax_min,220);
//    getCirqueUpAndDown(Hdist,rowUpMin_min,rowUpMax_max,rowDownMin_min,rowDownMax_max,20);

    dzlog_debug("rowUpMin_max == %d",rowUpMin_max);
    dzlog_debug("rowUpMax_min == %d",rowUpMax_min);
    dzlog_debug("rowDownMin_max == %d",rowDownMin_max);
    dzlog_debug("rowDownMax_min == %d",rowDownMax_min);

//    dzlog_debug("rowUpMin_min == %d",rowUpMin_min);
//    dzlog_debug("rowUpMax_max == %d",rowUpMax_max);
//    dzlog_debug("rowDownMin_min == %d",rowDownMin_min);
//    dzlog_debug("rowDownMax_max == %d",rowDownMax_max);
    dzlog_debug("外圆环高度 == %d",rowDownMax_min-rowDownMin_max);
    dzlog_debug("内圆环高度 == %d",rowUpMax_min-rowUpMin_max);
    fecr.outerCirque_innerRadius=outerRadius-(rowDownMax_min-rowDownMin_max);
    fecr.innerCirque_outerRadius=innerRadius+(rowUpMax_min-rowUpMin_max);

    cv::Mat imgRivetArea=imgcart(cv::Rect(0,rowUpMax_min,imgcart.cols,rowDownMin_max-rowUpMax_min));
    showImg("imgRivetArea",imgRivetArea);
    cv::Mat imgRivetArea_bin;
    threshold(imgRivetArea,imgRivetArea_bin,fecp.componentExistThreshold,255,CV_THRESH_BINARY);
    cv::Mat imgRivetArea_bin2=imgRivetArea_bin.clone();
    showImg("imgRivetArea_bin2",imgRivetArea_bin2);
    bwareaopen(imgRivetArea_bin,pow(fecp.rivetR,2));
    showImg("imgRivetArea_bin",imgRivetArea_bin);
    cv::Mat imgRivetArea_bin_bitwise;
    bitwise_not(imgRivetArea_bin,imgRivetArea_bin_bitwise);
    showImg("imgRivetArea_bin_bitwise",imgRivetArea_bin_bitwise);
    cCParam ccp;
    ccp.STAT_AREA=30000;
    ccp.STAT_WIDTH=300;
    ccp.STAT_HEIGHT=200;
    bwareaclose(imgRivetArea_bin_bitwise,ccp);
    showImg("imgRivetArea_bin_bitwise_bwclose",imgRivetArea_bin_bitwise);
    vector<Rect> rivetResults;
    bwareaopen(imgRivetArea_bin_bitwise,0.1*pow(fecp.rivetR,2),rivetResults);
    showImg("imgRivetArea_bin_bitwise_bwclose_open",imgRivetArea_bin_bitwise);
    cv::Mat imgshow;
    cvtColor(imgRivetArea,imgshow,CV_GRAY2BGR);
    for(int i=0;i<rivetResults.size();i++)
    {
        rectangle(imgshow,rivetResults[i],Scalar(0,255,0));
    }
    showImg("imgshow",imgshow);
    vector<Point> rivetCenters;
    cv::Point temp;
    for(int i=0;i<rivetResults.size();i++)
    {
        temp.x=rivetResults[i].x+rivetResults[i].width/2;
        temp.y=rivetResults[i].y+rivetResults[i].height/2;
        rivetCenters.push_back(temp);
    }
    int imgCartRivetYOffset=rowUpMax_min+rmin;
    int imgCartXOffset=0;
    getImgCartXOffset(fecp.angleRight,fecp.angleLeft,imgcart.cols,imgCartXOffset);
    vector<Point> rivetCentersOnWholeImgCart;
    points2WholeCartImg(rivetCenters,imgCartRivetYOffset,imgCartXOffset,rivetCentersOnWholeImgCart);
    vector<Point> rivetCentersOnImgSrc;
    pointsWholeCart2ImgSrc(rivetCentersOnWholeImgCart,outerCirqueCenter,outerRadius,fecp.preSetArea,rivetCentersOnImgSrc);
    cv::Mat imgshow2;
    cvtColor(fecp.imgSrc,imgshow2,CV_GRAY2BGR);
    for(int i=0;i<rivetCentersOnImgSrc.size();i++)
    {
        circle(imgshow2,rivetCentersOnImgSrc[i],fecp.rivetR,Scalar(0,0,255),2);
    }
    showImg("imgshow2",imgshow2);
    vector<Mat> rivetRectsInImgSrc;
    for(int i=0;i<rivetCentersOnImgSrc.size();i++)
    {
        cv::Mat tempmat;
        tempmat=fecp.imgSrc(cv::Rect(rivetCentersOnImgSrc[i].x-fecp.rivetR,rivetCentersOnImgSrc[i].y-fecp.rivetR,2*fecp.rivetR,2*fecp.rivetR));
        showImg("tempmat",tempmat);
        rivetRectsInImgSrc.push_back(tempmat);
    }
    fecr.rivetRadius=0;
    int index=0;
    for(int i=0;i<rivetRectsInImgSrc.size();i++)
    {
        index++;
        cv::Mat rivetBin;
        threshold(rivetRectsInImgSrc[i],rivetBin,fecp.componentExistThreshold,255,CV_THRESH_BINARY);
        cv::Mat img_rivet_preset_bin_robert;
        DVrobert(rivetBin,img_rivet_preset_bin_robert);
        showImg("img_rivet_preset_bin_robert",img_rivet_preset_bin_robert);
//        cv::Mat imgsobel;
////        Sobel(imgRivet_preset, imgsobel, imgRivet_preset.depth(), 1, 1, 3, 3, 1, BORDER_DEFAULT);
////        imshow("imgsobel",imgsobel);
////        imwrite("imgsobel.jpg",imgsobel);
//        DVsobel(imgRivet_preset_bin,imgsobel);
        vector<CircleParam> circles;
        houghCircle_DV(img_rivet_preset_bin_robert,1,0.01,40,50,0.9,5,10,circles);
        fecr.rivetRadius+=circles[0].radius;
        fecr.rivetCenters.push_back(circles[0].center);
    }
    // problem: index is zero
    if (index == 0)
    {
        dzlog_error("error!. Divided by zero.");
        return Err_CalibFrontEndFace;
    }
    else
    {
        fecr.rivetRadius=fecr.rivetRadius/index;
    }

    dzlog_debug("fecr.rivetRadius == %d",fecr.rivetRadius);
//    cv::Mat dist;
//    compare(imgRivetArea_bin, imgRivetArea_bin2, dist, CMP_NE);
//    showImg("imgdist",dist);

    // add character calibration
    calibCharacters(fecp, fecr);

    return Err_NoErr;
}

ErrCode FrontEndFace::detFrontEndFace(FrontEndFaceDetParam& efdp,FrontEndFaceDetResult& fedr)
{
    cv::Mat imgPreset=efdp.imgsrc(efdp.fecr.presetArea);
    showImg("imgPreset",imgPreset);
    float avgGrayLevel=0.0;
    float stdGreyLevel=0.0;
    //原图的平均灰度为55
    GetGrayAvgStdDev(efdp.imgsrc,avgGrayLevel,stdGreyLevel);
    dzlog_debug("avgGrayLevel == %f",avgGrayLevel);
    dzlog_debug("stdGreyLevel == %f",stdGreyLevel);
    if(avgGrayLevel < efdp.avgThreshold)//TODO:这里要查看实际工况
    {
        dzlog_error("图像的平均灰度小于亮度阈值,没有校准件");
        m_err=Err_CalibFrontEndFace;
        return m_err;
    }
    cv::Mat imgbin;
    dzlog_debug("efdp.detThreshold == %d",efdp.detThreshold);
    threshold(imgPreset,imgbin,efdp.detThreshold,255,CV_THRESH_BINARY_INV);
//    showImg("imgbin",imgbin);
    bwareaopen(imgbin,efdp.detCirqueCC);
    showImg("imgbin_afterbw",imgbin);

    vector<Point> outerCirquePoints;
    outerCirquePoints.clear();
//    cv::Mat outerbin=imgbin.clone();//TODO: FIXME:这里有一个大的问题还没有解决掉,后期要查看是什么原因引起的
    getEndFaceOuterRadiusPoints(imgbin,outerCirquePoints);
    dzlog_debug("outerCirquePoints.size() == %d",outerCirquePoints.size());

    float outerRadius=1300;//TODO:这里初始化了
    cv::Point outerCirqueCenter=cv::Point(0,0);
    if(!circleLeastFit(outerCirquePoints,outerCirqueCenter,outerRadius))
    {
        dzlog_error("最小二乘法拟合圆错误 ...");
        return Err_CircleLeastFit;
    }
    //TODO:这里的临时变量要改为fedr的结果变量
    fedr.outerCirque_outerRadius_center=outerCirqueCenter;
    fedr.outerCirque_outerRadius=outerRadius;
//    cout<<"outerCirqueCenter == "<<outerCirqueCenter<<endl;
    dzlog_debug("outerCirqueCenter.x==%d,y== %d",outerCirqueCenter.x,outerCirqueCenter.y);
    dzlog_debug("outerRadius == %f",outerRadius);
    if(outerRadius>efdp.fecr.outerRmaxThreshold || outerRadius<efdp.fecr.outerRminThreshold)
    {
        dzlog_error("outerRadius>efdp.outerRmaxThreshold|| outerRadius<efdp.outerRminThreshold ...");
    }
    sort(outerCirquePoints.begin(),outerCirquePoints.end(),points_x_sort);
    cv::Point pt1=outerCirquePoints[0];
    cv::Point pt2=outerCirquePoints[outerCirquePoints.size()-1];
    float angleLeft=atan2(outerCirqueCenter.y-pt1.y,pt1.x-outerCirqueCenter.x);
    float angleRight=atan2(outerCirqueCenter.y-pt2.y,pt2.x-outerCirqueCenter.x);
    fedr.angleLeft=angleLeft;
    fedr.angleRight=angleRight;
    dzlog_debug("angleleft == %f",angleLeft);
    dzlog_debug("angleRight == %f",angleRight);
    dzlog_debug("fedr.angleLeft == %f",fedr.angleLeft);
    if(angleLeft<angleRight)
    {
        dzlog_error("angleLeft<angleRight ...");
    }
    cv::Mat imgcart;
//    float thetaMin=CV_PI/3.0;
//    float thetaMax=2.0*CV_PI/3.0;

    //TODO:这里为什么要加偏移(需要弄明白)
    if(efdp.fecr.innerRminThreshold>2000)
    {
        dzlog_error("efdp.innerRminThreshold>2000 err detinnerCirque...");
        efdp.fecr.innerRminThreshold=700;//829
    }
    polar2cart(imgPreset, imgcart, outerCirqueCenter, outerRadius,efdp.fecp.angleRight,efdp.fecp.angleLeft,efdp.fecr.innerRminThreshold);
    showImg("imgcart",imgcart);

//    cv:Mat imgcart2;
//    polar2cart(imgPreset, imgcart2, outerCirqueCenter, outerRadius,0,2*CV_PI,0);
//    cv::imshow("imgcart2",imgcart2);
//    cv::waitKey(0);

    vector<Point> innerCirquePoints;
    innerCirquePoints.clear();
//    getEndFaceInnerRadiusPoints_horizontal(outerbin,innerCirquePoints);
    getEndFaceInnerRadiusPoints_vertical(imgbin,innerCirquePoints);
    dzlog_debug("innerCirquePoints.size() == %d",innerCirquePoints.size());

    float innerRadius=900;
    cv::Point innerCirqueCenter=cv::Point(0,0);
    if(!circleLeastFit(innerCirquePoints,innerCirqueCenter,innerRadius))
    {
        dzlog_error("最小二乘法拟合圆错误 ...");
        return Err_CircleLeastFit;
    }
    sort(innerCirquePoints.begin(),innerCirquePoints.end(),points_x_sort);
//    cv::Point pt3=innerCirquePoints[0];
//    cv::Point pt4=innerCirquePoints[innerCirquePoints.size()-1];
    dzlog_debug("innerCirqueCenter.x == %d,y == %d",innerCirqueCenter.x,innerCirqueCenter.y);
    dzlog_debug("innerRadius == %f",innerRadius);

    if(innerRadius>efdp.fecr.innerRmaxThreshold || innerRadius<efdp.fecr.innerRminThreshold)
    {
        dzlog_error("innerRadius>efdp.innerRmaxThreshold|| innerRadius<efdp.innerRminThreshold ...");
    }
    float centerDist=DVAlgorithm::getDistance(innerCirqueCenter,outerCirqueCenter);
    dzlog_debug("centerDist == %f",centerDist);
    if(centerDist>efdp.centerDistThreshold)
    {
        dzlog_error("centerDist>efdp.centerDistThreshold ...");
    }

    cv::Mat imgcartBin;
    threshold(imgcart,imgcartBin,efdp.detThreshold,255,CV_THRESH_BINARY_INV);
    bwareaopen(imgcartBin,efdp.detCirqueCC);
    showImg("imgcartBin",imgcartBin);
//    imwrite("imgcartbin.jpg",imgcartBin);

    cv::Mat imgcartbin_bitwise;
    bitwise_not(imgcartBin,imgcartbin_bitwise);
//    showImg("imgcartbin_bitwise",imgcartbin_bitwise);
    bwareaopen(imgcartbin_bitwise,efdp.detCirqueCC);//消除图像中间的铆钉和保持架的成像区域
    showImg("imgcartbin_bitwise2",imgcartbin_bitwise);
//    imwrite("imgcartbin_bitwise.jpg",imgcartbin_bitwise);
    int rowUpMin_max,rowUpMax_min,rowDownMin_max,rowDownMax_min;
    int rowUpMin_min,rowUpMax_max,rowDownMin_min,rowDownMax_max;
    cv::Mat Hdist = Mat::zeros(imgcartbin_bitwise.rows, 1, CV_32FC1);//CV_32FC1
    reduce(imgcartbin_bitwise, Hdist, 1, CV_REDUCE_SUM, CV_32FC1);//水平方向投影
    Hdist = Hdist / imgcart.cols;
//    cout<<Hdist<<endl;
    //这里根据不同的阈值设置,选出的有效区域的范围不同,
    getCirqueUpAndDown(Hdist,rowUpMin_max,rowUpMax_min,rowDownMin_max,rowDownMax_min,220);//TODO:这里的220后面可能是要开放出来的
    getCirqueUpAndDown(Hdist,rowUpMin_min,rowUpMax_max,rowDownMin_min,rowDownMax_max,20);//TODO:这里的20后面可能是要开放出来的
    float outerCirqueHeight=rowDownMax_min-rowDownMin_max;
    float innerCirqueHeight=rowUpMax_min-rowUpMin_max;
    if(outerCirqueHeight>efdp.fecr.outerCirqueHeightMax || outerCirqueHeight<efdp.fecr.outerCirqueHeightMin)
    {
        dzlog_error("outerCirqueHeight>efdp.outerCirqueHeightMax || outerCirqueHeight<efdp.outerCirqueHeightMin ...");
    }
    if(innerCirqueHeight>efdp.fecr.innerCirqueHeightMax || innerCirqueHeight<efdp.fecr.innerCirqueHeightMin)
    {
        dzlog_error("innerCirqueHeight>efdp.innerCirqueHeightMax || innerCirqueHeight<efdp.innerCirqueHeightMin");
    }
    cv::Mat imgRoiUp=imgcart(cv::Rect(0,rowUpMin_max,imgcart.cols,rowUpMax_min-rowUpMin_max));
    cv::Mat imgRoiDown=imgcart(cv::Rect(0,rowDownMin_max,imgcart.cols,rowDownMax_min-rowDownMin_max));
    showImg("imgRoiDown",imgRoiDown);
    showImg("imgRoiUp",imgRoiUp);
    cv::Mat imgRoiDownStdFiltDef;
    cv::Mat imgRoiUpStdFiltDef;

    //检测的方法
    //1.检测的图 2.检测核类型 3.检测的方差阈值  4.检测的连通域参数 5.平均灰度阈值 6.检测结果保存 7,返回的检测出缺陷的图片
    detBrightAndDarkSpot(imgRoiDown,1,15,efdp.endFaceDetCC,efdp.fecr.outerCirqueGreyThreshold,fedr,imgRoiDownStdFiltDef);//通过rowDownMin_max+efdp.innerRminThreshold 还原成原来的图坐标
    detBrightAndDarkSpot(imgRoiUp,0,15,efdp.endFaceDetCC,efdp.fecr.innerCirqueGreyThreshold,fedr,imgRoiUpStdFiltDef);
    //    cv::imshow("efdp.imgLabelTemplate",efdp.imgLabelTemplate);
    //    cv::waitKey(0);
    if(efdp.imgsrc.channels()==1)
    {
        cvtColor(efdp.imgsrc,fedr.imgResultShow,CV_GRAY2BGR);
    } else{
        fedr.imgResultShow=efdp.imgsrc;
    }

    getDefBoundingRect(imgRoiDownStdFiltDef,efdp,rowDownMin_max+efdp.fecr.innerRminThreshold,fedr);
    getDefBoundingRect(imgRoiUpStdFiltDef,efdp,rowUpMin_max+efdp.fecr.innerRminThreshold,fedr);
    showImg("imgresult",fedr.imgResultShow);

    //    imwrite("imgRoiDown.jpg",imgRoiDown);
    //    imwrite("imgRoiUp.jpg",imgRoiUp);


    fedr.imgDownRoi=imgRoiDown;
    fedr.imgUpRoi=imgRoiUp;
    fedr.imgRivetArea=imgcart(cv::Rect(0,rowUpMax_max,imgcart.cols,rowDownMin_min-rowUpMax_max));
//    showImg("fedr.imgRivetArea",fedr.imgRivetArea);
//    imwrite("imgRivetArea.jpg",fedr.imgRivetArea);
    //得到铆钉区域条中 铆钉中心位置的坐标点
    getCartRivetArea(fedr.imgRivetArea,efdp,fedr);

    //得到imgpreset图坐标下,铆钉位置的中心坐标
    dzlog_debug("rowUpMax_max == %d",rowUpMax_max);
    pointscart2polar(fedr.rivetCenters_cart,rowUpMax_max+efdp.fecr.innerRminThreshold,fedr);

    cv::Mat imgshow;
    if(imgPreset.channels()==1)
    {
        cvtColor(imgPreset,imgshow,CV_GRAY2BGR);
    }else{
        imgshow=imgPreset;
    }
    for(int i=0;i<fedr.rivetCenters_preset.size();i++)
    {
        cout<<fedr.rivetCenters_preset[i]<<endl;
        circle(imgshow,fedr.rivetCenters_preset[i],2,Scalar(0,0,255),2);
    }
    showImg("imgPresetxx",imgshow);
    getPresetRivetArea(efdp,fedr);


//模板匹配
//    labelTemplateMatching(fedr.imgUpRoi,efdp.imgLabelTemplate,fedr);
//数字检测
//    detDigital(imgRoiUpStdFiltDef,efdp,fedr);
//    imwrite("imgRoiUpStdFiltDef.jpg",imgRoiUpStdFiltDef);
//    vector<cv::Mat> digitals;
//    getCharactorsByCC(imgRoiUp,efdp.digDetCC,digitals);
    //接下来做标记及反变换
    return Err_NoErr;
}



ErrCode FrontEndFace::detFrontEndFace2(FrontEndFaceDetParam2& fedp,FrontEndFaceDetResult2& fedr)
{
    if(fedp.imgSrc.channels() == 1)
    {
        cvtColor(fedp.imgSrc,fedr.imgResult,CV_GRAY2BGR);
    }else{
        fedr.imgResult=fedp.imgSrc;
    }
    cv::Mat imgPreset=fedp.imgSrc(fedp.fecp.preSetArea);

    showImg("imgPreset",imgPreset);
    if(!checkIsComponentExist(imgPreset,fedp.fecp.componentExistThreshold,fedp.fecp.areaThreshold))
    {
        dzlog_error("注意查看工位是否有零件 ...");
        return Err_ComponentNotExist;
    }
    cv::Mat imgbin;
    threshold(imgPreset,imgbin,fedp.fecp.componentExistThreshold,255,CV_THRESH_BINARY);
    bwareaopen(imgbin,fecp_CC);
    bitwise_not(imgbin,imgbin);
//    threshold(imgPreset,imgbin,fedp.fecp.componentExistThreshold,255,CV_THRESH_BINARY_INV);
    showImg("imgbin",imgbin);
    bwareaopen(imgbin,fecp_CC);
    showImg("imgbin_afterbw",imgbin);

    vector<Point> outerCirquePoints;
    outerCirquePoints.clear();
    getEndFaceOuterRadiusPoints(imgbin,outerCirquePoints);
    for(int i=0;i<outerCirquePoints.size();i++)
    {
        cv::Point temp;
        temp.x=outerCirquePoints[i].x+fedp.fecp.preSetArea.x;
        temp.y=outerCirquePoints[i].y+fedp.fecp.preSetArea.y;
        circle(fedr.imgResult,temp,1,Scalar(0,255,0));
    }
    showImg("fedr.imgResult",fedr.imgResult);
    dzlog_debug("outerCirquePoints.size() == %d",outerCirquePoints.size());

    float outerRadius=0.0;
    cv::Point outerCirqueCenter=cv::Point(0,0);
//    sort(outerCirquePoints.begin(),outerCirquePoints.end(),points_y_sort);
//    if(abs(outerCirquePoints[0].y-outerCirquePoints[outerCirquePoints.size()-1].y) < 50)
//    {
//        dzlog_error("请调整光照或者调整调整二值化检测阈值 ...");
//        return Err_DetFrontEndFace;
//    }
    if(!circleLeastFit(outerCirquePoints,outerCirqueCenter,outerRadius))
    {
        dzlog_error("请调整光照或者调整调整二值化检测阈值 ...");
        return Err_CircleLeastFit;
    }
    dzlog_debug("outerCirqueCenter.x==%d,y== %d",outerCirqueCenter.x,outerCirqueCenter.y);
    dzlog_debug("outerRadius == %f",outerRadius);
    circle(fedr.imgResult,outerCirqueCenter+cv::Point(fedp.fecp.preSetArea.x,fedp.fecp.preSetArea.y),outerRadius,Scalar(0,0,255));
    showImg("fedr.imgResult",fedr.imgResult);
    if(DVAlgorithm::getDistance(fedp.fecr.outerRadiusCenter,outerCirqueCenter) > fedp.centerDist)
    {
        dzlog_error("检测过程中拟合的外圆圆心与校准的时候的圆心之间的距离过大 > %d",fedp.centerDist);
        return Err_DetFrontEndFace;
    }

    vector<Point> innerCirquePoints;
    innerCirquePoints.clear();
    getEndFaceInnerRadiusPoints_vertical(imgbin,innerCirquePoints);
    dzlog_debug("innerCirquePoints.size() == %d",innerCirquePoints.size());
    for(int i=0;i<innerCirquePoints.size();i++)
    {
        cv::Point temp;
        temp.x=innerCirquePoints[i].x+fedp.fecp.preSetArea.x;
        temp.y=innerCirquePoints[i].y+fedp.fecp.preSetArea.y;
        circle(fedr.imgResult,temp,1,Scalar(0,255,0));
    }
    showImg("fedr.imgResult",fedr.imgResult);
    if(innerCirquePoints.size() < 100)
    {
        dzlog_error("detInnerCirque Error ...");
    }
    float innerRadius=0.0;//初始化系数,这里是没用的
    cv::Point innerCirqueCenter=cv::Point(0,0);
//    sort(innerCirquePoints.begin(),innerCirquePoints.end(),points_y_sort);
//    if(abs(innerCirquePoints[0].y-innerCirquePoints[innerCirquePoints.size()-1].y) < 50)//TODO(hhg):这里要根据实际情况设定值,目前可能是设置的值偏大,为了防止所有的点都是平行的点
//    {
//        dzlog_error("请调整光照或者调整调整二值化检测阈值 ...");
//        return Err_DetReverseEndFace;
//    }
    if(!circleLeastFit(innerCirquePoints,innerCirqueCenter,innerRadius))
    {
        dzlog_error("请调整光照或者调整调整二值化检测阈值 ...");
        return Err_CircleLeastFit;
    }
    dzlog_debug("innerCirqueCenter.x == %d,y == %d",innerCirqueCenter.x,innerCirqueCenter.y);
    dzlog_debug("innerRadius == %f",innerRadius);
    if(DVAlgorithm::getDistance(fedp.fecr.innerRadiusCenter,innerCirqueCenter) > fedp.centerDist)
    {
        dzlog_error("检测过程中拟合的内圆圆心与校准的时候的圆心之间的距离过大 > %d",fedp.centerDist);
        return Err_DetFrontEndFace;
    }
    circle(fedr.imgResult,innerCirqueCenter+cv::Point(fedp.fecp.preSetArea.x,fedp.fecp.preSetArea.y),innerRadius,Scalar(0,0,255),2);
    showImg("fedr.imgResult",fedr.imgResult);


    cv::Mat imgcart;
//    float thetaMin=CV_PI/3.0;
//    float thetaMax=2.0*CV_PI/3.0;
    polar2cart(imgPreset, imgcart, outerCirqueCenter, outerRadius,fedp.fecp.angleRight,fedp.fecp.angleLeft,outerRadius*0.5);
    showImg("imgcart",imgcart);


    cv::Mat imgcartBin;
    threshold(imgcart,imgcartBin,fedp.fecp.componentExistThreshold,255,CV_THRESH_BINARY_INV);
    bwareaopen(imgcartBin,fecp_CC);
    showImg("imgcartBin",imgcartBin);
    cv::Mat imgcartbin_bitwise;
    bitwise_not(imgcartBin,imgcartbin_bitwise);
//    showImg("imgcartbin_bitwise",imgcartbin_bitwise);
    bwareaopen(imgcartbin_bitwise,fecp_CC);//消除图像中间的铆钉和保持架的成像区域
    showImg("imgcartbin_bitwise2",imgcartbin_bitwise);
//    imwrite("imgcartbin_bitwise.jpg",imgcartbin_bitwise);
    int rowUpMin_max,rowUpMax_min,rowDownMin_max,rowDownMax_min;
    int rowUpMin_min,rowUpMax_max,rowDownMin_min,rowDownMax_max;
    cv::Mat Hdist = Mat::zeros(imgcartbin_bitwise.rows, 1, CV_32FC1);//CV_32FC1
    reduce(imgcartbin_bitwise, Hdist, 1, CV_REDUCE_SUM, CV_32FC1);//水平方向投影
    Hdist = Hdist / imgcart.cols;
//    cout<<Hdist<<endl;
    //这里根据不同的阈值设置,选出的有效区域的范围不同,
    getCirqueUpAndDown(Hdist,rowUpMin_max,rowUpMax_min,rowDownMin_max,rowDownMax_min,220);//TODO:这里的220后面可能是要开放出来的
    getCirqueUpAndDown(Hdist,rowUpMin_min,rowUpMax_max,rowDownMin_min,rowDownMax_max,20);//TODO:这里的20后面可能是要开放出来的
    float outerCirqueHeight=rowDownMax_min-rowDownMin_max;
    float innerCirqueHeight=rowUpMax_min-rowUpMin_max;
    cv::Mat imgRoiUp=imgcart(cv::Rect(0,rowUpMin_max,imgcart.cols,rowUpMax_min-rowUpMin_max));//内圆
    cv::Mat imgRoiDown=imgcart(cv::Rect(0,rowDownMin_max,imgcart.cols,rowDownMax_min-rowDownMin_max));//外圆
    showImg("imgRoiUp",imgRoiUp);
    showImg("imgRoiDown",imgRoiDown);
    cv::Mat imgRoiUpStdFiltDef;
    cv::Mat imgRoiDownStdFiltDef;

    int imgCartXOffset=0;
    getImgCartXOffset(fedp.fecp.angleRight,fedp.fecp.angleLeft,imgcart.cols,imgCartXOffset);

    stdFilter(imgRoiUp,0,fedp.stdFilterThreshold,imgRoiUpStdFiltDef);
    showImg("imgRoiUpStdFiltDef",imgRoiUpStdFiltDef);
    vector<Rect> imgRoiUpDefRects;
    bwareaopen(imgRoiUpStdFiltDef,fedp.planeDefCC,imgRoiUpDefRects);
    vector<Point> innerPlaneDefPoints;
    getRectsBorderPoints(imgRoiUpDefRects,innerPlaneDefPoints);
    vector<Point> innerPlaneDefPoints_onWholeCart;
    points2WholeCartImg(innerPlaneDefPoints,outerRadius*0.5+rowUpMin_max,imgCartXOffset,innerPlaneDefPoints_onWholeCart);
    vector<Point> innerPlaneDefPoints_onImgSrc;
    pointsWholeCart2ImgSrc(innerPlaneDefPoints_onWholeCart,outerCirqueCenter,outerRadius,
                           fedp.fecp.preSetArea,innerPlaneDefPoints_onImgSrc);
    for(int i=0;i<innerPlaneDefPoints_onImgSrc.size();i++)
    {
        circle(fedr.imgResult,innerPlaneDefPoints_onImgSrc[i],1,Scalar(0,0,255),2);
    }
    showImg("fedr.imgResult",fedr.imgResult);


    stdFilter(imgRoiDown,0,fedp.stdFilterThreshold,imgRoiDownStdFiltDef);
    vector<Rect> imgRoiDownDefRects;
    bwareaopen(imgRoiDownStdFiltDef,fedp.planeDefCC,imgRoiDownDefRects);
    vector<Point> outerPlaneDefPoints;
    getRectsBorderPoints(imgRoiUpDefRects,outerPlaneDefPoints);
    vector<Point> outerPlaneDefPoints_onWholeCart;
    points2WholeCartImg(outerPlaneDefPoints,outerRadius*0.5+rowUpMin_max,imgCartXOffset,outerPlaneDefPoints_onWholeCart);
    vector<Point> outerPlaneDefPoints_onImgSrc;
    pointsWholeCart2ImgSrc(outerPlaneDefPoints_onWholeCart,outerCirqueCenter,outerRadius,
                           fedp.fecp.preSetArea,outerPlaneDefPoints_onImgSrc);
    for(int i=0;i<outerPlaneDefPoints_onImgSrc.size();i++)
    {
        circle(fedr.imgResult,outerPlaneDefPoints_onImgSrc[i],1,Scalar(0,0,255),2);
    }
    showImg("fedr.imgResult",fedr.imgResult);

    cv::Mat imgRivetArea=imgcart(cv::Rect(0,rowUpMax_min,imgcart.cols,rowDownMin_max-rowUpMax_min));
    showImg("imgRivetArea",imgRivetArea);
    cv::Mat imgRivetArea_bin;
    threshold(imgRivetArea,imgRivetArea_bin,fedp.fecp.componentExistThreshold,255,CV_THRESH_BINARY);
    cv::Mat imgRivetArea_bin2=imgRivetArea_bin.clone();
    showImg("imgRivetArea_bin2",imgRivetArea_bin2);
    bwareaopen(imgRivetArea_bin,pow(fedp.fecp.rivetR,2));
    showImg("imgRivetArea_bin",imgRivetArea_bin);
    cv::Mat imgRivetArea_bin_bitwise;
    bitwise_not(imgRivetArea_bin,imgRivetArea_bin_bitwise);
    showImg("imgRivetArea_bin_bitwise",imgRivetArea_bin_bitwise);
    cCParam ccp;
    ccp.STAT_AREA=30000;
    ccp.STAT_WIDTH=300;
    ccp.STAT_HEIGHT=200;
    bwareaclose(imgRivetArea_bin_bitwise,ccp);
    showImg("imgRivetArea_bin_bitwise_bwclose",imgRivetArea_bin_bitwise);
    vector<Rect> rivetResults;
    bwareaopen(imgRivetArea_bin_bitwise,0.1*pow(fedp.fecp.rivetR,2),rivetResults);
    showImg("imgRivetArea_bin_bitwise_bwclose_open",imgRivetArea_bin_bitwise);
    cv::Mat imgshow;
    cvtColor(imgRivetArea,imgshow,CV_GRAY2BGR);
    for(int i=0;i<rivetResults.size();i++)
    {
        rectangle(imgshow,rivetResults[i],Scalar(0,255,0));
    }
    showImg("imgshow",imgshow);
    vector<Point> rivetCenters;
    cv::Point temp;
    for(int i=0;i<rivetResults.size();i++)
    {
        temp.x=rivetResults[i].x+rivetResults[i].width/2;
        temp.y=rivetResults[i].y+rivetResults[i].height/2;
        rivetCenters.push_back(temp);
    }
    int imgCartRivetYOffset=rowUpMax_min+outerRadius*0.5;
    vector<Point> rivetCentersOnWholeImgCart;
    points2WholeCartImg(rivetCenters,imgCartRivetYOffset,imgCartXOffset,rivetCentersOnWholeImgCart);
    vector<Point> rivetCentersOnImgSrc;
    pointsWholeCart2ImgSrc(rivetCentersOnWholeImgCart,outerCirqueCenter,outerRadius,fedp.fecp.preSetArea,rivetCentersOnImgSrc);
    cv::Mat imgshow2;
    cvtColor(fedp.imgSrc,imgshow2,CV_GRAY2BGR);
    for(int i=0;i<rivetCentersOnImgSrc.size();i++)
    {
        circle(imgshow2,rivetCentersOnImgSrc[i],fedp.fecp.rivetR,Scalar(0,0,255),2);
    }
    showImg("imgshow2",imgshow2);
    vector<Mat> rivetRectsInImgSrc;
    for(int i=0;i<rivetCentersOnImgSrc.size();i++)
    {
        cv::Mat tempmat;
        tempmat=fedp.imgSrc(cv::Rect(rivetCentersOnImgSrc[i].x-fedp.fecp.rivetR,rivetCentersOnImgSrc[i].y-fedp.fecp.rivetR,2*fedp.fecp.rivetR,2*fedp.fecp.rivetR));
        showImg("tempmat",tempmat);
        rivetRectsInImgSrc.push_back(tempmat);
    }
    int index=0;
    for(int i=0;i<rivetRectsInImgSrc.size();i++)
    {
        index++;
        cv::Mat rivetBin;
        threshold(rivetRectsInImgSrc[i],rivetBin,fedp.fecp.componentExistThreshold,255,CV_THRESH_BINARY);
        cv::Mat img_rivet_preset_bin_robert;
        DVrobert(rivetBin,img_rivet_preset_bin_robert);
        showImg("img_rivet_preset_bin_robert",img_rivet_preset_bin_robert);
//        cv::Mat imgsobel;
////        Sobel(imgRivet_preset, imgsobel, imgRivet_preset.depth(), 1, 1, 3, 3, 1, BORDER_DEFAULT);
////        imshow("imgsobel",imgsobel);
////        imwrite("imgsobel.jpg",imgsobel);
//        DVsobel(imgRivet_preset_bin,imgsobel);
        vector<CircleParam> circles;
        houghCircle_DV(img_rivet_preset_bin_robert,1,0.01,40,50,0.9,5,10,circles);
        cv::Point rivetCenterTemp(rivetCentersOnImgSrc[i].x-fedp.fecp.rivetR,rivetCentersOnImgSrc[i].y-fedp.fecp.rivetR);
        if(circles[0].radius<fedp.rivetRThreshold)
        {
            circle(fedr.imgResult,circles[0].center+rivetCenterTemp,circles[0].radius,Scalar(0,0,255),2);
            dzlog_error("铆钉铆合不好,现在检测到的半径为 == %f",circles[0].radius);
            return Err_DetFrontEndFace;
        }else{
            circle(fedr.imgResult,circles[0].center+rivetCenterTemp,circles[0].radius,Scalar(0,255,0),2);
        }
    }

    showImg("fedr.imgResult",fedr.imgResult);

    // add new method
    find_starting_point = detCharacters(fedp, fedr, ith_image);
    cout << "find_starting_point = " << find_starting_point << endl;
    cout << "ith_image = " << ith_image << endl;

////模板匹配
////    labelTemplateMatching(fedr.imgUpRoi,fedp.imgLabelTemplate,fedr);
////数字检测
////    detDigital(imgRoiUpStdFiltDef,fedp,fedr);
////    imwrite("imgRoiUpStdFiltDef.jpg",imgRoiUpStdFiltDef);
////    vector<cv::Mat> digitals;
////    getCharactorsByCC(imgRoiUp,fedp.digDetCC,digitals);
//    //接下来做标记及反变换
//    return Err_NoErr;
}



ErrCode FrontEndFace::saveFrontEndfaceCalibResult(FrontEndFaceCalibrateResult fecr,string filename,string xmlRootname)
{
    createXML(filename.c_str(),xmlRootname.c_str());
    XMLDocument doc;
    int res=doc.LoadFile(filename.c_str());
    if(res!=0)
    {
        dzlog_error("load xml file failed");
        return Err_SaveXml;
    }

    XMLElement *root=doc.RootElement();

    XMLElement *fecrnode=doc.NewElement("FrontEndFaceCalibrateResult");
    root->InsertEndChild(fecrnode);

    XMLElement *presetAreaX=doc.NewElement("presetAreaX");
    XMLElement *presetAreaY=doc.NewElement("presetAreaY");
    XMLElement *presetAreaWidth=doc.NewElement("presetAreaWidth");
    XMLElement *presetAreaHeight=doc.NewElement("presetAreaHeight");
    fecrnode->InsertEndChild(presetAreaX);
    fecrnode->InsertEndChild(presetAreaY);
    fecrnode->InsertEndChild(presetAreaWidth);
    fecrnode->InsertEndChild(presetAreaHeight);
    XMLText *texpresetAreaX = doc.NewText(to_string(fecr.presetArea.x).c_str());
    XMLText *texpresetAreaY = doc.NewText(to_string(fecr.presetArea.y).c_str());
    XMLText *texpresetAreaWidth = doc.NewText(to_string(fecr.presetArea.width).c_str());
    XMLText *texpresetAreaHeight = doc.NewText(to_string(fecr.presetArea.height).c_str());
    presetAreaX->InsertEndChild(texpresetAreaX);
    presetAreaY->InsertEndChild(texpresetAreaY);
    presetAreaWidth->InsertEndChild(texpresetAreaWidth);
    presetAreaHeight->InsertEndChild(texpresetAreaHeight);

    XMLElement *innerRadiusCenterX=doc.NewElement("innerRadiusCenterX");
    XMLElement *innerRadiusCenterY=doc.NewElement("innerRadiusCenterY");
    fecrnode->InsertEndChild(innerRadiusCenterX);
    fecrnode->InsertEndChild(innerRadiusCenterY);
    XMLText *texInnerRadiusCenterX = doc.NewText(to_string(fecr.innerRadiusCenter.x).c_str());
    XMLText *texInnerRadiusCenterY = doc.NewText(to_string(fecr.innerRadiusCenter.y).c_str());
    innerRadiusCenterX->InsertEndChild(texInnerRadiusCenterX);
    innerRadiusCenterY->InsertEndChild(texInnerRadiusCenterY);

    XMLElement *outerRadiusCenterX=doc.NewElement("outerRadiusCenterX");
    XMLElement *outerRadiusCenterY=doc.NewElement("outerRadiusCenterY");
    fecrnode->InsertEndChild(outerRadiusCenterX);
    fecrnode->InsertEndChild(outerRadiusCenterY);
    XMLText *texOuterRadiusCenterX = doc.NewText(to_string(fecr.outerRadiusCenter.x).c_str());
    XMLText *texOuterRadiusCenterY = doc.NewText(to_string(fecr.outerRadiusCenter.y).c_str());
    outerRadiusCenterX->InsertEndChild(texOuterRadiusCenterX);
    outerRadiusCenterY->InsertEndChild(texOuterRadiusCenterY);

    //最外径
    XMLElement *outerCirque_outerRadius=doc.NewElement("outerCirque_outerRadius");
    fecrnode->InsertEndChild(outerCirque_outerRadius);
    XMLText *texOuterRadius1 = doc.NewText(to_string(fecr.outerCirque_outerRadius).c_str());
    outerCirque_outerRadius->InsertEndChild(texOuterRadius1);

    XMLElement *outerCirque_innerRadius=doc.NewElement("outerCirque_innerRadius");
    fecrnode->InsertEndChild(outerCirque_innerRadius);
    XMLText *texouterCirque_innerRadius = doc.NewText(to_string(fecr.outerCirque_innerRadius).c_str());
    outerCirque_innerRadius->InsertEndChild(texouterCirque_innerRadius);

    XMLElement *innerCirque_innerRadius=doc.NewElement("innerCirque_innerRadius");
    fecrnode->InsertEndChild(innerCirque_innerRadius);
    XMLText *texinnerCirque_innerRadius = doc.NewText(to_string(fecr.innerCirque_innerRadius).c_str());
    innerCirque_innerRadius->InsertEndChild(texinnerCirque_innerRadius);

    XMLElement *innerCirque_outerRadius=doc.NewElement("innerCirque_outerRadius");
    fecrnode->InsertEndChild(innerCirque_outerRadius);
    XMLText *texinnerCirque_outerRadius = doc.NewText(to_string(fecr.innerCirque_outerRadius).c_str());
    innerCirque_outerRadius->InsertEndChild(texinnerCirque_outerRadius);

    XMLElement *outerCirqueHeight=doc.NewElement("outerCirqueHeight");
    fecrnode->InsertEndChild(outerCirqueHeight);
    XMLText *texouterCirqueHeight = doc.NewText(to_string(fecr.outerCirqueHeight).c_str());
    outerCirqueHeight->InsertEndChild(texouterCirqueHeight);

    XMLElement *innerCirqueHeight=doc.NewElement("innerCirqueHeight");
    fecrnode->InsertEndChild(innerCirqueHeight);
    XMLText *texinnerCirqueHeight = doc.NewText(to_string(fecr.innerCirqueHeight).c_str());
    innerCirqueHeight->InsertEndChild(texinnerCirqueHeight);

    XMLElement *outerCirqueGreyThreshold=doc.NewElement("outerCirqueGreyThreshold");
    fecrnode->InsertEndChild(outerCirqueGreyThreshold);
    XMLText *texouterCirqueGreyThreshold = doc.NewText(to_string(fecr.outerCirqueGreyThreshold).c_str());
    outerCirqueGreyThreshold->InsertEndChild(texouterCirqueGreyThreshold);

    XMLElement *innerCirqueGreyThreshold=doc.NewElement("innerCirqueGreyThreshold");
    fecrnode->InsertEndChild(innerCirqueGreyThreshold);
    XMLText *texinnerCirqueGreyThreshold = doc.NewText(to_string(fecr.innerCirqueGreyThreshold).c_str());
    innerCirqueGreyThreshold->InsertEndChild(texinnerCirqueGreyThreshold);

    XMLElement *centerDist=doc.NewElement("centerDist");
    fecrnode->InsertEndChild(centerDist);
    XMLText *texcenterDist = doc.NewText(to_string(fecr.centerDist).c_str());
    centerDist->InsertEndChild(texcenterDist);

    XMLElement *outerCirqueHeightMin=doc.NewElement("outerCirqueHeightMin");
    fecrnode->InsertEndChild(outerCirqueHeightMin);
    XMLText *texouterCirqueHeightMin = doc.NewText(to_string(fecr.outerCirqueHeightMin).c_str());
    outerCirqueHeightMin->InsertEndChild(texouterCirqueHeightMin);

    XMLElement *outerCirqueHeightMax=doc.NewElement("outerCirqueHeightMax");
    fecrnode->InsertEndChild(outerCirqueHeightMax);
    XMLText *texouterCirqueHeightMax = doc.NewText(to_string(fecr.outerCirqueHeightMax).c_str());
    outerCirqueHeightMax->InsertEndChild(texouterCirqueHeightMax);

    XMLElement *innerCirqueHeightMin=doc.NewElement("innerCirqueHeightMin");
    fecrnode->InsertEndChild(innerCirqueHeightMin);
    XMLText *texinnerCirqueHeightMin = doc.NewText(to_string(fecr.innerCirqueHeightMin).c_str());
    innerCirqueHeightMin->InsertEndChild(texinnerCirqueHeightMin);

    XMLElement *innerCirqueHeightMax=doc.NewElement("innerCirqueHeightMax");
    fecrnode->InsertEndChild(innerCirqueHeightMax);
    XMLText *texinnerCirqueHeightMax = doc.NewText(to_string(fecr.innerCirqueHeightMax).c_str());
    innerCirqueHeightMax->InsertEndChild(texinnerCirqueHeightMax);

    XMLElement *innerRminThreshold=doc.NewElement("innerRminThreshold");
    fecrnode->InsertEndChild(innerRminThreshold);
    XMLText *texinnerRminThreshold = doc.NewText(to_string(fecr.innerRminThreshold).c_str());
    innerRminThreshold->InsertEndChild(texinnerRminThreshold);

    XMLElement *innerRmaxThreshold=doc.NewElement("innerRmaxThreshold");
    fecrnode->InsertEndChild(innerRmaxThreshold);
    XMLText *texinnerRmaxThreshold = doc.NewText(to_string(fecr.innerRmaxThreshold).c_str());
    innerRmaxThreshold->InsertEndChild(texinnerRmaxThreshold);

    XMLElement *outerRminThreshold=doc.NewElement("outerRminThreshold");
    fecrnode->InsertEndChild(outerRminThreshold);
    XMLText *texouterRminThreshold = doc.NewText(to_string(fecr.outerRminThreshold).c_str());
    outerRminThreshold->InsertEndChild(texouterRminThreshold);

    XMLElement *outerRmaxThreshold=doc.NewElement("outerRmaxThreshold");
    fecrnode->InsertEndChild(outerRmaxThreshold);
    XMLText *texouterRmaxThreshold = doc.NewText(to_string(fecr.outerRmaxThreshold).c_str());
    outerRmaxThreshold->InsertEndChild(texouterRmaxThreshold);

    XMLElement *detThreshold=doc.NewElement("detThreshold");
    fecrnode->InsertEndChild(detThreshold);
    XMLText *texdetThreshold = doc.NewText(to_string(fecr.detThreshold).c_str());
    detThreshold->InsertEndChild(texdetThreshold);

    doc.SaveFile(filename.c_str());
    return Err_NoErr;
}

ErrCode FrontEndFace::saveFrontEndfaceCalibResult2(FrontEndFaceCalibrateParam2 fecp,FrontEndFaceCalibrateResult2 fecr,string filename,string xmlRootName)
{
    createXML(filename.c_str(),xmlRootName.c_str());
    XMLDocument doc;
    int res=doc.LoadFile(filename.c_str());
    if(res!=0)
    {
        dzlog_error("load xml file failed");
        return Err_SaveXml;
    }

    XMLElement *root=doc.RootElement();

    XMLElement *fecrnode=doc.NewElement("FrontEndFaceCalibrateResult");
    root->InsertEndChild(fecrnode);
    XMLElement *preSetArea=doc.NewElement("preSetArea");
    fecrnode->InsertEndChild(preSetArea);
    XMLElement *preSetAreaX=doc.NewElement("X");
    XMLElement *preSetAreaY=doc.NewElement("Y");
    XMLElement *preSetAreaWidth=doc.NewElement("Width");
    XMLElement *preSetAreaHeight=doc.NewElement("Height");
    preSetArea->InsertEndChild(preSetAreaX);
    preSetArea->InsertEndChild(preSetAreaY);
    preSetArea->InsertEndChild(preSetAreaWidth);
    preSetArea->InsertEndChild(preSetAreaHeight);
    XMLText *tex_preSetAreaX = doc.NewText(to_string(fecp.preSetArea.x).c_str());
    XMLText *tex_preSetAreaY = doc.NewText(to_string(fecp.preSetArea.y).c_str());
    XMLText *tex_preSetAreaWidth = doc.NewText(to_string(fecp.preSetArea.width).c_str());
    XMLText *tex_preSetAreaHeight = doc.NewText(to_string(fecp.preSetArea.height).c_str());
    preSetAreaX->InsertEndChild(tex_preSetAreaX);
    preSetAreaY->InsertEndChild(tex_preSetAreaY);
    preSetAreaWidth->InsertEndChild(tex_preSetAreaWidth);
    preSetAreaHeight->InsertEndChild(tex_preSetAreaHeight);

    //int componentExistThreshold=100;//判断零件是否存在的阈值,这个阈值也作为选取零件区域的阈值
    //    int areaThreshold=preSetArea.width*preSetArea.height*0.1;
    //    float angleLeft=2.18;
    //    float angleRight=0.96;
    //    int rivetR=80;

    XMLElement *componentExistThreshold=doc.NewElement("componentExistThreshold");
    fecrnode->InsertEndChild(componentExistThreshold);
    XMLText *tex_componentExistThreshold = doc.NewText(to_string(fecp.componentExistThreshold).c_str());
    componentExistThreshold->InsertEndChild(tex_componentExistThreshold);
    
    XMLElement *areaThreshold=doc.NewElement("areaThreshold");
    fecrnode->InsertEndChild(areaThreshold);
    XMLText *tex_areaThreshold = doc.NewText(to_string(fecp.areaThreshold).c_str());
    areaThreshold->InsertEndChild(tex_areaThreshold);

    XMLElement *angleLeft=doc.NewElement("angleLeft");
    fecrnode->InsertEndChild(angleLeft);
    XMLText *tex_angleLeft = doc.NewText(to_string(fecp.angleLeft).c_str());
    angleLeft->InsertEndChild(tex_angleLeft);

    XMLElement *angleRight=doc.NewElement("angleRight");
    fecrnode->InsertEndChild(angleRight);
    XMLText *tex_angleRight = doc.NewText(to_string(fecp.angleRight).c_str());
    angleRight->InsertEndChild(tex_angleRight);

    XMLElement *rivetR=doc.NewElement("rivetR");
    fecrnode->InsertEndChild(rivetR);
    XMLText *tex_rivetR = doc.NewText(to_string(fecp.rivetR).c_str());
    rivetR->InsertEndChild(tex_rivetR);

    
    //    cv::Point innerRadiusCenter;    //内圆心
    //    cv::Point outerRadiusCenter;    //外圆心
    //

    XMLElement* innerRadiusCenter=doc.NewElement("innerRadiusCenter");
    fecrnode->InsertEndChild(innerRadiusCenter);
    XMLElement* innerRadiusCenter_x=doc.NewElement("X");
    XMLElement* innerRadiusCenter_y=doc.NewElement("Y");
    innerRadiusCenter->InsertEndChild(innerRadiusCenter_x);
    innerRadiusCenter->InsertEndChild(innerRadiusCenter_y);
    XMLText* tex_innerRadiusCenter_x=doc.NewText(to_string(fecr.innerRadiusCenter.x).c_str());
    XMLText* tex_innerRadiusCenter_y=doc.NewText(to_string(fecr.innerRadiusCenter.y).c_str());
    innerRadiusCenter_x->InsertEndChild(tex_innerRadiusCenter_x);
    innerRadiusCenter_y->InsertEndChild(tex_innerRadiusCenter_y);

    XMLElement* outerRadiusCenter=doc.NewElement("outerRadiusCenter");
    fecrnode->InsertEndChild(outerRadiusCenter);
    XMLElement* outerRadiusCenter_x=doc.NewElement("X");
    XMLElement* outerRadiusCenter_y=doc.NewElement("Y");
    outerRadiusCenter->InsertEndChild(outerRadiusCenter_x);
    outerRadiusCenter->InsertEndChild(outerRadiusCenter_y);
    XMLText* tex_outerRadiusCenter_x=doc.NewText(to_string(fecr.outerRadiusCenter.x).c_str());
    XMLText* tex_outerRadiusCenter_y=doc.NewText(to_string(fecr.outerRadiusCenter.y).c_str());
    outerRadiusCenter_x->InsertEndChild(tex_outerRadiusCenter_x);
    outerRadiusCenter_y->InsertEndChild(tex_outerRadiusCenter_y);

    //    int outerCirque_outerRadius;    //外圆环外径 主要判别依据
    //    int outerCirque_innerRadius;    //外圆环内径 目前是根据极坐标转换出来的图横向投影得到的边界值与上面边界的一个差值得到的
    //    int innerCirque_innerRadius;    //内圆环内径 主要判别依据
    //    int innerCirque_outerRadius;    //内圆环外径
    //    int rivetRadius;                //正常铆合铆钉的半径大小
    //    vector<Point> rivetCenters;     //铆钉圆心
    XMLElement *outerCirque_outerRadius=doc.NewElement("outerCirque_outerRadius");
    fecrnode->InsertEndChild(outerCirque_outerRadius);
    XMLText *tex_outerCirque_outerRadius = doc.NewText(to_string(fecr.outerCirque_outerRadius).c_str());
    outerCirque_outerRadius->InsertEndChild(tex_outerCirque_outerRadius);

    XMLElement *outerCirque_innerRadius=doc.NewElement("outerCirque_innerRadius");
    fecrnode->InsertEndChild(outerCirque_innerRadius);
    XMLText *tex_outerCirque_innerRadius = doc.NewText(to_string(fecr.outerCirque_innerRadius).c_str());
    outerCirque_innerRadius->InsertEndChild(tex_outerCirque_innerRadius);

    XMLElement *innerCirque_innerRadius=doc.NewElement("innerCirque_innerRadius");
    fecrnode->InsertEndChild(innerCirque_innerRadius);
    XMLText *tex_innerCirque_innerRadius = doc.NewText(to_string(fecr.innerCirque_innerRadius).c_str());
    innerCirque_innerRadius->InsertEndChild(tex_innerCirque_innerRadius);

    XMLElement *innerCirque_outerRadius=doc.NewElement("innerCirque_outerRadius");
    fecrnode->InsertEndChild(innerCirque_outerRadius);
    XMLText *tex_innerCirque_outerRadius = doc.NewText(to_string(fecr.innerCirque_outerRadius).c_str());
    innerCirque_outerRadius->InsertEndChild(tex_innerCirque_outerRadius);

    XMLElement *rivetRadius=doc.NewElement("rivetRadius");
    fecrnode->InsertEndChild(rivetRadius);
    XMLText *tex_rivetRadius = doc.NewText(to_string(fecr.rivetRadius).c_str());
    rivetRadius->InsertEndChild(tex_rivetRadius);

    XMLElement *rivetCenters_size=doc.NewElement("rivetCenters_size");
    fecrnode->InsertEndChild(rivetCenters_size);
    XMLText *tex_rivetCenters_size = doc.NewText(to_string(fecr.rivetCenters.size()).c_str());
    rivetCenters_size->InsertEndChild(tex_rivetCenters_size);
    char rivetCenter[20];
    for(int i=0;i<fecr.rivetCenters.size();i++)
    {
        sprintf(rivetCenter,"rivetCenter%d",i);
        XMLElement* rivetCenters=doc.NewElement(rivetCenter);
        fecrnode->InsertEndChild(rivetCenters);
        XMLElement* rivetCenters_x=doc.NewElement("X");
        XMLElement* rivetCenters_y=doc.NewElement("Y");
        rivetCenters->InsertEndChild(rivetCenters_x);
        rivetCenters->InsertEndChild(rivetCenters_y);
        XMLText* tex_rivetCenters_x=doc.NewText(to_string(fecr.rivetCenters[i].x).c_str());
        XMLText* tex_rivetCenters_y=doc.NewText(to_string(fecr.rivetCenters[i].y).c_str());
        rivetCenters_x->InsertEndChild(tex_rivetCenters_x);
        rivetCenters_y->InsertEndChild(tex_rivetCenters_y);
    }
    doc.SaveFile(filename.c_str());
    return Err_NoErr;
}

ErrCode FrontEndFace::parseFrontEndfaceCalibResult(string xmlname,FrontEndFaceDetParam& efdp)
{
    //cv::Point innerRadiusCenter;//圆心内
    //    cv::Point outerRadiusCenter;//圆心外
    //    float outerCirque_outerRadius;//外圆环外径,主要判别依据
    //    float outerCirque_innerRadius;//外圆环内径 目前是根据极坐标转换出来的图横向投影得到的边界值与上面边界的一个差值得到的
    //    float innerCirque_innerRadius;//内圆环内径,主要判别依据
    //    float innerCirque_outerRadius;//内圆环外径
    //    int outerCirqueGreyThreshold;//外圆环的灰度
    //    int innerCirqueGreyThreshold;//内圆环的灰度
    //    float centerDistThreshold;//最内最外之间圆环的圆心之间的距离阈值 如果两个圆心之间的距离大于这个阈值,则报错
    //
    //    int detThreshold;//二值化检测边界的阈值 这个是根据前面标准的时候得到的外圆环和内圆环的平均灰度+一个偏移阈值  起始也可以用这个作为检测二值化的阈值参数
    //

    XMLDocument doc;
    int res=doc.LoadFile(xmlname.c_str());
    if(res!=0)
    {
        dzlog_error("load xml file failed");
        return Err_ParseXml;
    }

    XMLElement *root=doc.RootElement();
    XMLElement* fecrnode=root->FirstChildElement("FrontEndFaceCalibrateResult");

    XMLElement* presetAreaX=fecrnode->FirstChildElement("presetAreaX");
    efdp.fecr.presetArea.x=atoi(presetAreaX->GetText());

    XMLElement* presetAreaY=fecrnode->FirstChildElement("presetAreaY");
    efdp.fecr.presetArea.y=atoi(presetAreaY->GetText());

    XMLElement* presetAreaWidth=fecrnode->FirstChildElement("presetAreaWidth");
    efdp.fecr.presetArea.width=atoi(presetAreaWidth->GetText());

    XMLElement* presetAreaHeight=fecrnode->FirstChildElement("presetAreaHeight");
    efdp.fecr.presetArea.height=atoi(presetAreaHeight->GetText());

    XMLElement* innerRadiusCenterX=fecrnode->FirstChildElement("innerRadiusCenterX");
    XMLElement* innerRadiusCenterY=fecrnode->FirstChildElement("innerRadiusCenterY");
    efdp.fecr.innerRadiusCenter.x=atoi(innerRadiusCenterX->GetText());//将字符串为int 类型
    efdp.fecr.innerRadiusCenter.y=atoi(innerRadiusCenterY->GetText());//将字符串为int 类型

    XMLElement* outerRadiusCenterX=fecrnode->FirstChildElement("outerRadiusCenterX");
    XMLElement* outerRadiusCenterY=fecrnode->FirstChildElement("outerRadiusCenterY");
    efdp.fecr.outerRadiusCenter.x=atoi(outerRadiusCenterX->GetText());
    efdp.fecr.outerRadiusCenter.y=atoi(outerRadiusCenterY->GetText());

    XMLElement* outerCirque_outerRadius=fecrnode->FirstChildElement("outerCirque_outerRadius");
    efdp.fecr.outerCirque_outerRadius=atof(outerCirque_outerRadius->GetText());

    XMLElement* outerCirque_innerRadius=fecrnode->FirstChildElement("outerCirque_innerRadius");
    efdp.fecr.outerCirque_innerRadius=atof(outerCirque_innerRadius->GetText());

    XMLElement* innerCirque_innerRadius=fecrnode->FirstChildElement("innerCirque_innerRadius");
    efdp.fecr.innerCirque_innerRadius=atof(innerCirque_innerRadius->GetText());

    XMLElement* innerCirque_outerRadius=fecrnode->FirstChildElement("innerCirque_outerRadius");
    efdp.fecr.innerCirque_outerRadius=atof(innerCirque_outerRadius->GetText());

    XMLElement* outerCirqueGreyThreshold=fecrnode->FirstChildElement("outerCirqueGreyThreshold");
    efdp.fecr.outerCirqueGreyThreshold=atoi(outerCirqueGreyThreshold->GetText());

    XMLElement* innerCirqueGreyThreshold=fecrnode->FirstChildElement("innerCirqueGreyThreshold");
    efdp.fecr.innerCirqueGreyThreshold=atoi(innerCirqueGreyThreshold->GetText());

//    XMLElement* centerDistThreshold=fecrnode->FirstChildElement("centerDistThreshold");
//    efdp.centerDistThreshold=atof(centerDistThreshold->GetText());

    XMLElement* detThreshold=fecrnode->FirstChildElement("detThreshold");
    efdp.detThreshold=atoi(detThreshold->GetText());

    XMLElement* outerCirqueHeightMin=fecrnode->FirstChildElement("outerCirqueHeightMin");
    efdp.fecr.outerCirqueHeightMin=atof(outerCirqueHeightMin->GetText());

    XMLElement* outerCirqueHeightMax=fecrnode->FirstChildElement("outerCirqueHeightMax");
    efdp.fecr.outerCirqueHeightMax=atof(outerCirqueHeightMax->GetText());

    XMLElement* innerCirqueHeightMin=fecrnode->FirstChildElement("innerCirqueHeightMin");
    efdp.fecr.innerCirqueHeightMin=atof(innerCirqueHeightMin->GetText());

    XMLElement* innerCirqueHeightMax=fecrnode->FirstChildElement("innerCirqueHeightMax");
    efdp.fecr.innerCirqueHeightMax=atof(innerCirqueHeightMax->GetText());

    XMLElement* innerRminThreshold=fecrnode->FirstChildElement("innerRminThreshold");
    efdp.fecr.innerRminThreshold=atof(innerRminThreshold->GetText());

    XMLElement* innerRmaxThreshold=fecrnode->FirstChildElement("innerRmaxThreshold");
    efdp.fecr.innerRmaxThreshold=atof(innerRmaxThreshold->GetText());

    XMLElement* outerRminThreshold=fecrnode->FirstChildElement("outerRminThreshold");
    efdp.fecr.outerRminThreshold=atof(outerRminThreshold->GetText());

    XMLElement* outerRmaxThreshold=fecrnode->FirstChildElement("outerRmaxThreshold");
    efdp.fecr.outerRmaxThreshold=atof(outerRmaxThreshold->GetText());

    return Err_NoErr;
}

ErrCode FrontEndFace::parseFrontEndfaceCalibResult2(string xmlname,FrontEndFaceDetParam2& fedp)
{
    XMLDocument doc;
    int res=doc.LoadFile(xmlname.c_str());
    if(res!=0)
    {
        dzlog_error("load xml file failed");
        return Err_ParseXml;
    }

    XMLElement *root=doc.RootElement();
    XMLElement* fecrnode=root->FirstChildElement("FrontEndFaceCalibrateResult");

    XMLElement* preSetArea=fecrnode->FirstChildElement("preSetArea");
    XMLElement* preSetArea_x=preSetArea->FirstChildElement("X");
    XMLElement* preSetArea_y=preSetArea->FirstChildElement("Y");
    XMLElement* preSetArea_width=preSetArea->FirstChildElement("Width");
    XMLElement* preSetArea_height=preSetArea->FirstChildElement("Height");
    fedp.fecp.preSetArea.x=atoi(preSetArea_x->GetText());
    fedp.fecp.preSetArea.y=atoi(preSetArea_y->GetText());
    fedp.fecp.preSetArea.width=atoi(preSetArea_width->GetText());
    fedp.fecp.preSetArea.height=atoi(preSetArea_height->GetText());

    XMLElement* componentExistThreshold=fecrnode->FirstChildElement("componentExistThreshold");
    fedp.fecp.componentExistThreshold=atoi(componentExistThreshold->GetText());

    XMLElement* areaThreshold=fecrnode->FirstChildElement("areaThreshold");
    fedp.fecp.areaThreshold=atoi(areaThreshold->GetText());

    XMLElement* angleLeft=fecrnode->FirstChildElement("angleLeft");
    fedp.fecp.angleLeft=atof(angleLeft->GetText());

    XMLElement* angleRight=fecrnode->FirstChildElement("angleRight");
    fedp.fecp.angleRight=atof(angleRight->GetText());

    XMLElement* rivetR=fecrnode->FirstChildElement("rivetR");
    fedp.fecp.rivetR=atoi(rivetR->GetText());

    XMLElement* innerRadiusCenter=fecrnode->FirstChildElement("innerRadiusCenter");
    XMLElement* innerRadiusCenter_x=innerRadiusCenter->FirstChildElement("X");
    XMLElement* innerRadiusCenter_y=innerRadiusCenter->FirstChildElement("Y");
    fedp.fecr.innerRadiusCenter.x=atoi(innerRadiusCenter_x->GetText());
    fedp.fecr.innerRadiusCenter.y=atoi(innerRadiusCenter_y->GetText());

    XMLElement* outerRadiusCenter=fecrnode->FirstChildElement("outerRadiusCenter");
    XMLElement* outerRadiusCenter_x=outerRadiusCenter->FirstChildElement("X");
    XMLElement* outerRadiusCenter_y=outerRadiusCenter->FirstChildElement("Y");
    fedp.fecr.outerRadiusCenter.x=atoi(outerRadiusCenter_x->GetText());
    fedp.fecr.outerRadiusCenter.y=atoi(outerRadiusCenter_y->GetText());

    XMLElement* outerCirque_outerRadius=fecrnode->FirstChildElement("outerCirque_outerRadius");
    fedp.fecr.outerCirque_outerRadius=atoi(outerCirque_outerRadius->GetText());

    XMLElement* outerCirque_innerRadius=fecrnode->FirstChildElement("outerCirque_innerRadius");
    fedp.fecr.outerCirque_innerRadius=atoi(outerCirque_innerRadius->GetText());

    XMLElement* innerCirque_innerRadius=fecrnode->FirstChildElement("innerCirque_innerRadius");
    fedp.fecr.innerCirque_innerRadius=atoi(innerCirque_innerRadius->GetText());

    XMLElement* innerCirque_outerRadius=fecrnode->FirstChildElement("innerCirque_outerRadius");
    fedp.fecr.innerCirque_outerRadius=atoi(innerCirque_outerRadius->GetText());

    XMLElement* rivetRadius=fecrnode->FirstChildElement("rivetRadius");
    fedp.fecr.rivetRadius=atoi(rivetRadius->GetText());

    //这里没有用的
    XMLElement* rivetCenters_size=fecrnode->FirstChildElement("rivetCenters_size");
    int rivetCenterSize=atoi(rivetCenters_size->GetText());
    char rivetCenterName[20];
    for(int i=0;i<rivetCenterSize;i++)
    {
        cv::Point temp(0,0);
        sprintf(rivetCenterName,"rivetCenter%d",i);
        XMLElement* outerRadiusCenter=fecrnode->FirstChildElement(rivetCenterName);
        XMLElement* outerRadiusCenter_x=outerRadiusCenter->FirstChildElement("X");
        XMLElement* outerRadiusCenter_y=outerRadiusCenter->FirstChildElement("Y");
        temp.x=atoi(outerRadiusCenter_x -> GetText());
        temp.y=atoi(outerRadiusCenter_y -> GetText());
        fedp.fecr.rivetCenters.push_back(temp);
    }
    return Err_NoErr;
}

//竖向扫描 得到外圆边界的坐标点集合
void FrontEndFace::getEndFaceOuterRadiusPoints(cv::Mat& imgbin,std::vector<Point>& points)
{
    for(int i=0;i<imgbin.cols;i++)
    {
        for(int j=0;j<imgbin.rows;j++)
        {
            if(imgbin.ptr<uchar>(j)[i]<20)
            {
                points.push_back(cv::Point(i,j));
                break;
            }
        }
    }
}




//横向中点向左向右扫描 这样扫描扫描到的点数会比较少
void FrontEndFace::getEndFaceInnerRadiusPoints_horizontal(cv::Mat& imgbin,std::vector<Point>& points)
{
    dzlog_debug("get start getEndFaceInnerRadiusPoints_horizontal ...");
    int mid=imgbin.cols/2;
    for(int i=imgbin.rows-1;i>0;i--)
    {
//        dzlog_debug("imgbin.ptr<uchar>(i)[mid] == %d",imgbin.ptr<uchar>(i)[mid]);
        if(imgbin.ptr<uchar>(i)[mid] > 0){
            for(int j=mid;j>0;j--)
            {
                if(imgbin.ptr<uchar>(i)[j]<20)
                {
                    points.push_back(cv::Point(j,i));
                    break;
                }
            }
            for(int k=mid;k<imgbin.cols-1;k++)
            {
                if(imgbin.ptr<uchar>(i)[k]<20)
                {
                    points.push_back(cv::Point(k,i));
                    break;
                }
            }
        }else{
            break;
        }
    }
}

void FrontEndFace::getEndFaceInnerRadiusPoints_vertical(cv::Mat& imgbin,std::vector<Point>& points)
{
//    showImg("imgbin_vertical",imgbin);
    dzlog_debug("get start getEndFaceInnerRadiusPoints_vertical");
    int mid=imgbin.cols/2;
    for(int i=mid;i>0;i--)
    {
        if(imgbin.ptr<uchar>(imgbin.rows-1)[i]>100)
        {
            for(int j=imgbin.rows-1;j>0;j--)
            {
                if(imgbin.ptr<uchar>(j)[i]<20)
                {
                    points.push_back(cv::Point(i,j));
                    break;
                }
            }
        }else{
            break;
        }
    }

    for(int i=mid;i<imgbin.cols-1;i++)
    {
        if(imgbin.ptr<uchar>(imgbin.rows-1)[i]>100)
        {
            for(int j=imgbin.rows-1;j>0;j--)
            {
                if(imgbin.ptr<uchar>(j)[i]<20)
                {
                    points.push_back(cv::Point(i,j));
                    break;
                }
            }
        } else{
            break;
        }
    }
}




/**
 * 得到imgcartbin图像横向投影后,分别得到中间两条环状区域的上边界和下边界
 * @param imgcartbin 极坐标二值化后的图
 * @param rowUpMin 极坐标上面的圆环的上边界
 * @param rowUpMax 极坐标上面的圆环的下边界
 * @param rowDownMin 下面的圆环的上边界
 * @param rowDownMax 下面的圆环的下边界
 * @param binLightThreshold 水平投影的平均灰度 的阈值,该阈值用于确定上面边界的范围情况
 */
void FrontEndFace::getCirqueUpAndDown(cv::Mat& Hdist,int& rowUpMin,int& rowUpMax,int& rowDownMin,int& rowDownMax,int binLightThreshold)
{

//    std::cout << "Hdist == " << Hdist << endl;
    rowUpMin=Hdist.rows/2;//最内侧的环形区域
    rowUpMax=0;
    rowDownMin=Hdist.rows;
    rowDownMax=Hdist.rows/2;
    for (int i=Hdist.rows/2;i>0;i--)
    {
        if (Hdist.at<float>(i, 0) > binLightThreshold)
        {
            if(rowUpMin>i){
                rowUpMin=i;
            }
            if(rowUpMax<i)
            {
                rowUpMax=i;
            }
        }
    }

    for(int i = Hdist.rows/2; i < Hdist.rows; i++)
    {
        if(Hdist.at<float>(i, 0) > binLightThreshold)
        {
            if(rowDownMin>i)
            {
                rowDownMin=i;
            }
            if(rowDownMax<i)
            {
                rowDownMax=i;
            }
        }
    }
}


/**
 * 检测环的亮斑和暗斑
 * @param imgcartRoi 待检测的ROI图片
 * @param kernelType 核类型
 * @param stdfilterThreshold 方差阈值 目前30
 * @param endFaceDetCC 连通域的大小参数
 * @param avgThreshold
 * @param fedr 检测结果
 */
void FrontEndFace::detBrightAndDarkSpot(cv::Mat& imgcartRoi,int kernelType,int stdfilterThreshold,cCParam endFaceDetCC,float avgThreshold,FrontEndFaceDetResult& fedr,cv::Mat& imgdist)
{
    //TODO:采用方差加均值的方法求出目标区域,检测亮斑的时候(方差+均值)要大于一定的值  检测暗斑的时候(-方差+均值)要小于一定的值
    float a=0.2;
    float b=0.3;
    MatrixXf imgsrc_matrix(imgcartRoi.rows, imgcartRoi.cols);
    cv2eigen(imgcartRoi,imgsrc_matrix);
    imgsrc_matrix/=255;
//    cout<<imgsrc_matrix<<endl;
    eigen2cv(imgsrc_matrix,imgcartRoi);
    Mat mask;
    Mat mask1;
    Eigen::MatrixXf xx;
    float n;
    if(kernelType == 0){
        mask=Mat::ones (3,3,CV_8UC1);
        xx = MatrixXf::Ones(3, 3);
    }else if(kernelType == 1)
    {
        mask=Mat::ones (3,5,CV_8UC1);
        xx = MatrixXf::Ones(3, 5);
    }else if(kernelType==2)
    {
        mask=Mat::ones (3,7,CV_8UC1);
        xx = MatrixXf::Ones(3, 7);
    }else{
        mask=Mat::ones (3,9,CV_8UC1);
        xx = MatrixXf::Ones(3, 9);
    }
    n=xx.sum();
    xx/=n;
//    cout<<xx<<endl;

    eigen2cv(xx, mask1);
    cout<<mask1<<endl;
    cv::Mat avg;
    cv::filter2D(imgcartRoi,avg,-1,mask1,Point(-1,-1),0,BORDER_DEFAULT);
//    cout<<"avg == "<<avg<<endl;
//    imshow("avg",avg);
//    cv::waitKey(0);
//    imwrite("filter2D.jpg",avg*255);
//    cv::Mat imgblurout;
//    blur(imgcartRoi, imgblurout, Size(3,3), Point(-1,-1), BORDER_DEFAULT );
//    cout<<"imgblurout == "<<imgblurout<<endl;
//    cv::imshow("imgblurout",imgblurout);
//    cv::waitKey(0);
//    imwrite("imgblurout.jpg",imgblurout*255);
//    cv::Mat minusimg=imgblurout-avg;
//    cout<<"minusimg == "<<minusimg<<endl;
    cv::Mat c1,c2,c3,mean;
    cv::filter2D(imgcartRoi.mul(imgcartRoi),c1,-1,mask1,Point(-1,-1),0,BORDER_REFLECT);
//    cout<<c1<<endl;
    cv::filter2D(imgcartRoi,mean,-1,mask,Point(-1,-1),0,BORDER_REFLECT);
//    cout<<mean<<endl;
    c2 = mean.mul(mean)/(n*n);
//    cout<<c2<<endl;
    cv::Mat temp;
    temp=max((c1-c2),0);
    cv::Mat J;
    cv::sqrt(temp,J);
    J*=255;
    //将数据打印输出到文件中
//    ofstream outFile;//创建了一个ofstream 对象
//    outFile.open("information.txt");//outFile 与一个文本文件关联
//    outFile<<J;    //小数点格式显示double
//    cout<<J<<endl;
    J=J>stdfilterThreshold;
    showImg("J",J);
//    cv::Mat imgdist;
    vector<Rect> defRectResults;
    bwareaopen(J,imgdist,endFaceDetCC,defRectResults);
//    cout<<J<<endl;
    showImg("stdfilt",imgdist);
//    bwareaopen(J,lefParam.outerWhiteCcp);
    dzlog_debug("defRectResults.size() == %d",defRectResults.size());
    for(int i=0;i<defRectResults.size();i++)
    {
        //宽度小于STAT_WIDTH  缺陷起始位置在STAT_HEIGHT_EDGE 上面  缺陷的高度不够
        dzlog_debug("defRectResults[%d].width == %d",i,defRectResults[i].width);
        dzlog_debug("defRectResults[%d].y == %d",i,defRectResults[i].y);
        dzlog_debug("defRectResults[%d].height == %d",i,defRectResults[i].height);
        if(defRectResults[i].width<endFaceDetCC.STAT_WIDTH ||
           defRectResults[i].height<endFaceDetCC.STAT_HEIGHT)
        {
            imgdist(defRectResults[i])=0;
            continue;
        }else{
            cv::Mat imgdefRoi=imgcartRoi(defRectResults[i]);
            showImg("imgdefRoi",imgdefRoi);
            cv::Mat mean;
            cv::Mat stdDev;
            cv::meanStdDev(imgdefRoi, mean, stdDev);
            dzlog_debug("mean.ptr<double>(0)[0] == %f ",mean.ptr<double>(0)[0]);
            if(mean.ptr<double>(0)[0]>avgThreshold)//这里可以是正常标定的图的平均灰度
            {
                fedr.hasDef= true;
                fedr.deft=BrightSpot;
                //TODO:检出的缺陷标记
            }else{
                fedr.hasDef= true;
                fedr.deft=DarkSpot;
            }
        }
    }
}


cv::Point FrontEndFace::ptChangeToSrcImg(FrontEndFaceDetParam& efdp,cv::Point pointRoi)
{
    cv::Point tmp;
    tmp.x = pointRoi.x + efdp.fecr.presetArea.x;
    tmp.y = pointRoi.y + efdp.fecr.presetArea.y;
    return tmp;
}

void FrontEndFace::getDefBoundingRect(cv::Mat& imgdist,FrontEndFaceDetParam efdp,float imgcartOFFSET,FrontEndFaceDetResult& fedr)
{
    //得到缺陷的最小包围矩形
    showImg("imgdistfinal",imgdist);
    cv::Mat imgstddistshow;
    cvtColor(imgdist,imgstddistshow,CV_GRAY2BGR);
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    cv::findContours(imgdist, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));
    std::vector<Rect> rectVec;
    std::sort(contours.begin(), contours.end(), contoursSort);
    cv::RotatedRect minRect;
    Rect boundrect;
    vector<Rect> boundRects;
    Point2f rect[4];
    for(int i=0;i<contours.size();i++)
    {
        if(contours[i].size()>20)//TODO:缺陷的外边缘点数要大于某一个值
        {
            minRect=minAreaRect(Mat(contours[i]));
            minRect.points(rect);//把最小外界矩形的4个点赋值给rect
            dzlog_debug("min(abs(getDistance(rect[0],rect[1])),abs(getDistance(rect[1],rect[2])) == %f",
                        min(abs(DVAlgorithm::getDistance(rect[0],rect[1])),abs(DVAlgorithm::getDistance(rect[1],rect[2]))));
            if(min(abs(DVAlgorithm::getDistance(rect[0],rect[1])),abs(DVAlgorithm::getDistance(rect[1],rect[2]))) < efdp.endFaceDetCC.STAT_WIDTH)
            {
                continue;
            }else{
                boundrect=boundingRect(Mat(contours[i]));
                cv::rectangle(imgstddistshow, boundrect, Scalar(0, 0, 255),1, LINE_8,0);
                boundRects.push_back(boundrect);
            }
        }
    }
    showImg("stddistshow",imgstddistshow);

    vector<Point> defPoints;
    for(int i=0;i<boundRects.size();i++)
    {
        for(int j=boundRects[i].x;j<boundRects[i].x+boundRects[i].width;j++)
        {
            defPoints.push_back(cv::Point(j,boundRects[i].y+imgcartOFFSET));//TODO:这里每个y后面都要加上一个原始对应的偏移值
            defPoints.push_back(cv::Point(j,boundRects[i].y+boundRects[i].height+imgcartOFFSET));
        }
        for(int k=boundRects[i].y;k<boundRects[i].y+boundRects[i].height;k++)
        {
            defPoints.push_back(cv::Point(boundRects[i].x,k+imgcartOFFSET));
            defPoints.push_back(cv::Point(boundRects[i].x+boundRects[i].width,k+imgcartOFFSET));
        }
    }
    vector<Point> defPointsAtPresetImg;
    dzlog_debug("defPoints.size() == %d",defPoints.size());
    if(defPoints.size()>30){//TODO:缺陷点的个数大于某一个值
        pointCart2polar(defPoints, defPointsAtPresetImg, fedr.outerCirque_outerRadius_center, fedr.outerCirque_outerRadius,fedr.angleRight+CV_PI);
        //TODO:转换回原始图的关系
    }

    //在原图上画缺陷标记
    dzlog_debug("defPointsAtPresetImg.size() == %d",defPointsAtPresetImg.size());

    for(int i=0;i<defPointsAtPresetImg.size();i++)
    {
//        dzlog_debug("ptChangeToSrcImg(efdp,defPointsAtPresetImg[%d]).x ==%d,y==%d",i,ptChangeToSrcImg(efdp,defPointsAtPresetImg[i]).x,ptChangeToSrcImg(efdp,defPointsAtPresetImg[i]).y);
        circle(fedr.imgResultShow,ptChangeToSrcImg(efdp,defPointsAtPresetImg[i]),1,Scalar(0,255,255),2);
    }
}

//TODO:这里得到的imgStdfilt是通过求局部方差的方式得到的缺陷检测区域的图片,  用于数字的检测可以采用直接二值化的方法求 目前采用局部方差求解出的有两条边缘特征线,若采用a*s+b*x 的方法求,效果会好一点
void FrontEndFace::detDigital(cv::Mat& imgStdfilt,FrontEndFaceDetParam efdp,FrontEndFaceDetResult& fedr)
{
    cv::Mat imgDet=cv::Mat::zeros(imgStdfilt.rows,imgStdfilt.cols,CV_8UC1);
    for(int i=0;i<imgDet.rows;i++)
    {
        for(int j=0;j<imgDet.cols;j++)
        {
            imgDet.ptr<uchar>(i)[j]=imgStdfilt.ptr<uchar>(imgStdfilt.rows-i-1)[j];
        }
    }
    showImg("imgDet",imgDet);
    cv::Mat imgDetClose;
//    cv::Mat kernel=cv::getStructuringElement();
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(imgDet, imgDetClose, MORPH_CLOSE, element);
//    imshow("imgDetClose",imgDetClose);
    cv::waitKey(0);
    vector<Rect> digitalRectResults;
    cv::Mat imgdistbw;
    bwareaopen(imgDetClose,imgdistbw,efdp.digDetCC,digitalRectResults);
    std::sort(digitalRectResults.begin(),digitalRectResults.end(),rect_x_sort);
//    imshow("imgDetClosebw",imgdistbw);
    cv::waitKey(0);
    char name[20];
    for(int i=0;i<digitalRectResults.size();i++)
    {
        //宽度小于STAT_WIDTH  缺陷起始位置在STAT_HEIGHT_EDGE 上面  缺陷的高度不够
        dzlog_debug("defRectResults[i].width == %d",digitalRectResults[i].width);
        dzlog_debug("defRectResults[i].y == %d",digitalRectResults[i].y);
        dzlog_debug("defRectResults[i].height == %d",digitalRectResults[i].height);
        if(digitalRectResults[i].width<efdp.digDetCC.STAT_WIDTH ||
           digitalRectResults[i].height<efdp.digDetCC.STAT_HEIGHT)
        {
            imgdistbw(digitalRectResults[i])=0;
            continue;
        }else{
            if(digitalRectResults[i].area()>efdp.labelDetCC.STAT_AREA && digitalRectResults[i].width>efdp.labelDetCC.STAT_WIDTH)
            {
                //这里是打印的标记的提取区域
                cv::Mat imgLabelROI=imgdistbw(digitalRectResults[i]);
                cv::Mat imgUpRoiFlip;
                flip(fedr.imgUpRoi,imgUpRoiFlip,0);
                cv::Mat imgLabelSrc=imgUpRoiFlip(digitalRectResults[i]);
//                sprintf(name,"%d_imgLabel.jpg");
//                imwrite(name,imgLabelSrc);

//                imshow("imgLabelROI",imgLabelROI);
//                waitKey(0);
            }else{
                //这里是数字的提取区域//TODO:这里可以根据选取到的地方进行模板匹配
                cv::Mat imgDigitalROI=imgdistbw(digitalRectResults[i]);
//                imshow("imgDigitalROI",imgDigitalROI);
//                waitKey(0);
            }

//            imshow("imgDigitalROI",imgDigitalROI);
//            waitKey(0);
//            sprintf(name,"%d.jpg",i);
//            imwrite(name,imgDigitalROI);
            //TODO:这里要增加每个检测的数字 检测算法 ,目前的检测算法有以下两种思路,1.用模板匹配的方法 2.用深度学习mnist数据集训练的方法
        }
    }
}

//实验证明模板匹配灰度图和RGB的图是一样的
//打印标签的模板匹配
void FrontEndFace::labelTemplateMatching(cv::Mat& imgUpRoi,cv::Mat& imgtemplate,FrontEndFaceDetResult& fedr)
{
//    cv::imshow("imgUpRoi",imgUpRoi);
//    cv::imshow("imgtemplate",imgtemplate);
    cv::waitKey(0);
    cout<<imgUpRoi.channels()<<endl;
//    cout<<imgtemplate.channels()<<endl;
    //enum { TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCORR=2, TM_CCORR_NORMED=3, TM_CCOEFF=4, TM_CCOEFF_NORMED=5 };
    //平方差匹配 TM_SQDIFF 利用平方差进行匹配,最好的匹配为0,匹配越差,匹配值越大
    //标准平方差匹配 TM_SQDIFF_NORMED
    //相关匹配 TM_CCORR 采用模板和图像间的乘法操作,所以较大的数表示匹配程度较高,0表示最坏的匹配效果
    //标准相关匹配 TM_CCORR_NORMED
    //相关匹配 TM_CCOEFF 这类方法将模板对其均值的相对值与图像对其均值的相关值进行匹配,1表示完美匹配,-1表示糟糕的匹配,0表示没有任何相关性(随机序列)
    //标准相关匹配 TM_CCOEFF_NORMED
    Mat result;
//    int result_cols = imgUpRoi.cols - imgtemplate.cols + 1;
//    int result_rows = imgUpRoi.rows - imgtemplate.rows + 1;
//    if(result_cols < 0 || result_rows < 0)
//    {
//        dzlog_error("Please input correct image!");
//        return;
//    }
//    result.create(result_cols, result_rows, CV_32F);
//    cout<<imgUpRoi.depth()<<endl;
//    dzlog_debug("imgUpRoi.depth() == %d",imgUpRoi.depth());
    cv::Mat imgUpRoi_tem;
//    cv::Mat imgTemplate_tem;
    cvtColor(imgUpRoi,imgUpRoi_tem,CV_GRAY2BGR);
//    cvtColor(imgtemplate,imgTemplate_tem,CV_GRAY2BGR);
//    cv::imshow("imgUpRoi_tem",imgUpRoi_tem);
//    cv::imshow("imgTemplate_tem",imgTemplate_tem);
//    cv::waitKey(0);
    cv::Mat imgTemplateGray;
    cvtColor(imgtemplate,imgTemplateGray,CV_BGR2GRAY);
    matchTemplate(imgUpRoi,imgTemplateGray, result, 5);
    double minVal = -1;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;
//    cout << "匹配度minVal：" << minVal << endl;
//    cout << "匹配度maxVal：" << maxVal << endl;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    dzlog_debug("匹配度minVal： == %f",minVal);
    dzlog_debug("匹配度maxVal： == %f",maxVal);
//    cout << "匹配度minVal：" << minVal << endl;
//    cout << "匹配度maxVal：" << maxVal << endl;
    //取大值(视匹配方法而定)
//    matchLoc = minLoc;
    matchLoc = maxLoc;
    Mat mask;
    cvtColor(imgUpRoi,mask,CV_GRAY2BGR);
    //绘制最匹配的区域
    rectangle(mask, matchLoc, Point(matchLoc.x + imgtemplate.cols, matchLoc.y + imgtemplate.rows), Scalar(0, 255, 0), 2, 8, 0);
//    imshow("mask",mask);
    cv::waitKey(0);
}

//得到长方形的铆钉的位置区域
void FrontEndFace::getCartRivetArea(cv::Mat& imgRivetArea,FrontEndFaceDetParam efdp,FrontEndFaceDetResult& fedr)
{
    cv::Mat imgbin;
    cv::threshold(imgRivetArea,imgbin,60,255,CV_THRESH_BINARY);
    showImg("imgbin",imgbin);
    cv::Mat bwdist;
    vector<Rect> results;
    bwareaopen(imgbin,bwdist,efdp.rivetDetCC,results);
    showImg("bwdist",bwdist);
    dzlog_debug("results.size() == %d",results.size());
    std::sort(results.begin(),results.end(),rect_x_sort);
    for(int i=0;i<results.size();i++)
    {
        //宽度小于STAT_WIDTH  缺陷起始位置在STAT_HEIGHT_EDGE 上面  缺陷的高度不够
        dzlog_debug("results[%d].width == %d",i,results[i].width);
        dzlog_debug("results[%d].y == %d",i,results[i].y);
        dzlog_debug("results[%d].height == %d",i,results[i].height);
        if(results[i].width<efdp.rivetDetCC.STAT_WIDTH ||
           results[i].height<efdp.rivetDetCC.STAT_HEIGHT)
        {
            imgRivetArea(results[i])=0;
            continue;
        }else{
            if(results[i].area()>100 && results[i].width>100 && results[i].height>100)//TODO这里的检测约束的参数要开放出来的
            {
                if(bwdist.ptr<uchar>(results[i].y+results[i].height/2)[results[i].x+results[i].width/2] <10)
                {
                    cv::Mat imgrivetCut=bwdist(results[i]);
                    showImg("imgrivetCut",imgrivetCut);
                    cv::Point imgrivetCutCenter=cv::Point(results[i].x+results[i].width/2,results[i].y+results[i].height/2);
                    fedr.rivetImgs_cart.push_back(imgrivetCut);
                    fedr.rivetCenters_cart.push_back(imgrivetCutCenter);
                }
            }
        }
    }
}

void FrontEndFace::detRivetArea(FrontEndFaceDetParam efdp,FrontEndFaceDetResult& fedr)
{
    for(int i=0;i<fedr.rivetImgs_cart.size();i++)
    {
        cv::Mat imgbin;
        cv::threshold(fedr.rivetImgs_cart[i],imgbin,50,255,CV_THRESH_BINARY);
//        imshow("imgbin",imgbin);
//        cv::waitKey(0);
    }
}

void FrontEndFace::pointscart2polar(vector<Point>& ploarPoints,float imgcartOFFSET,FrontEndFaceDetResult& fedr)
{
    vector<Point> rivetCentersTemp;
    Point ptTemp;
    for(int i=0;i<ploarPoints.size();i++)
    {
        ptTemp.x=ploarPoints[i].x;
        ptTemp.y=ploarPoints[i].y+imgcartOFFSET;
        dzlog_debug("ptTemp.x == %d",ptTemp.x);
        dzlog_debug("ptTemp.y == %d",ptTemp.y);
        rivetCentersTemp.push_back(ptTemp);
    }
    dzlog_debug("fedr.angleRight == %f",fedr.angleRight);
    pointCart2polar(rivetCentersTemp, fedr.rivetCenters_preset, fedr.outerCirque_outerRadius_center, fedr.outerCirque_outerRadius,fedr.angleRight+CV_PI);

    dzlog_debug("fedr.rivetCenters_preset.size() == %d",fedr.rivetCenters_preset.size());
}

void FrontEndFace::getPresetRivetArea(FrontEndFaceDetParam efdp,FrontEndFaceDetResult& fedr)
{
//    char name[20];
    cv::Rect rectTemp;
    cv::Mat imgRivet_preset;
    cv::Mat imgpreset=efdp.imgsrc(efdp.fecr.presetArea);
    dzlog_debug("fedr.rivetCenters_cart.size() == %d",fedr.rivetCenters_preset.size());
    for(int i=0;i<fedr.rivetCenters_preset.size();i++)
    {
        if(fedr.rivetCenters_preset[i].x+2*efdp.rivetCutR>imgpreset.cols || fedr.rivetCenters_preset[i].y+2*efdp.rivetCutR > imgpreset.rows)
        {
            continue;
        }
        rectTemp=cv::Rect((fedr.rivetCenters_preset[i].x-efdp.rivetCutR)>0?fedr.rivetCenters_preset[i].x-efdp.rivetCutR:0,
                          fedr.rivetCenters_preset[i].y-efdp.rivetCutR>0?fedr.rivetCenters_preset[i].y-efdp.rivetCutR:0,
                          2*efdp.rivetCutR,2*efdp.rivetCutR);
        imgRivet_preset=imgpreset(rectTemp);
        showImg("imgRivet_preset",imgRivet_preset);
//        sprintf(name,"%d_imgRivet_preset.jpg",i);
//        imwrite(name,imgRivet_preset);
        cv::Mat imgRivet_preset_bin;
        threshold(imgRivet_preset,imgRivet_preset_bin,70,255,CV_THRESH_BINARY_INV);
        showImg("imgRivet_preset_bin",imgRivet_preset_bin);
//        bwareaclose(imgRivet_preset_bin,efdp.rivet_Det_close_CC);
//        showImg("bw_imgRivet_preset_bin",imgRivet_preset_bin);


        cv::Mat img_rivet_preset_bin_robert;
        DVrobert(imgRivet_preset_bin,img_rivet_preset_bin_robert);
        showImg("img_rivet_preset_bin_robert",img_rivet_preset_bin_robert);
//        cv::Mat imgsobel;
////        Sobel(imgRivet_preset, imgsobel, imgRivet_preset.depth(), 1, 1, 3, 3, 1, BORDER_DEFAULT);
////        imshow("imgsobel",imgsobel);
////        imwrite("imgsobel.jpg",imgsobel);
//        DVsobel(imgRivet_preset_bin,imgsobel);
        vector<CircleParam> circles;
        houghCircle_DV(img_rivet_preset_bin_robert,1,0.01,40,50,0.9,5,10,circles);
//        cv::imshow("imgRivet_preset",imgRivet_preset);
//        cv::waitKey(0);

        if(circles.size()>0)
        {
            //对所有houghCircle检测到的圆进行排序,半径最大的排前面
            std::sort(circles.begin(),circles.end(),circle_R_sort);
            dzlog_debug("circles[0].radius == %f",circles[0].radius);
            if(circles[0].radius<30)//TODO:这里是铆钉处拟合圆的大小,如果小于设定的阈值,则会报铆钉缺失的问题 这里标定的时候要确定该值
            {
                dzlog_error("铆合不好...");
            }
        }
    }
}

void FrontEndFace::getCharactorsByCC(cv::Mat& img,int detThreshold,cCParam digCC,cCParam digBigCC,vector<Rect>& digitalRects,vector<Mat>& digitalImgs)
{
    cv::Mat imgbin;
    threshold(img,imgbin,detThreshold,255,CV_THRESH_BINARY_INV);
    showImg("imgbinDigital",imgbin);
    vector<Rect> digtalRectsTemp;
    bwareaopen(imgbin,digCC,digtalRectsTemp);
    cv::Mat imgTemp;
    for(int i=0;i<digtalRectsTemp.size();i++)
    {
        if(digtalRectsTemp[i].area()<digBigCC.STAT_AREA && digtalRectsTemp[i].width<digBigCC.STAT_WIDTH&&digtalRectsTemp[i].height<digBigCC.STAT_HEIGHT)
        {
            digitalRects.push_back(digtalRectsTemp[i]);
            imgTemp=img(digtalRectsTemp[i]);
            digitalImgs.push_back(imgTemp);
        }
    }
}


void FrontEndFace::detBrightSpot(cv::Mat& imgsrc,int detThreshold,cCParam defDetCC,vector<Rect>& defRects)
{
    cv::Mat imgbin;
    threshold(imgsrc,imgbin,detThreshold,255,CV_THRESH_BINARY);
    showImg("imgbin_detBrightSpot",imgbin);
    bwareaopen(imgbin,defDetCC,defRects);
}

void FrontEndFace::detDarkSpot(cv::Mat& imgsrc,int detThreshold,cCParam defDetCC,vector<Rect>& defRects)
{
    cv::Mat imgbin;
    threshold(imgsrc,imgbin,detThreshold,255,CV_THRESH_BINARY_INV);
    showImg("imgbin_detDarkSpot",imgbin);
    bwareaopen(imgbin,defDetCC,defRects);
}

void FrontEndFace::calibCharacters(FrontEndFaceCalibrateParam2& fefcp, FrontEndFaceCalibrateResult2& fefcr)
{
    // get image and crop image
    cv::Mat img_preset = fefcp.imgSrc(fefcp.preSetArea);

    // binarize the image
    cv::Mat img_bin;
    threshold(img_preset, img_bin, fefcp.componentExistThreshold, 255, THRESH_BINARY_INV);

    // get the center and radius
//    Point outer_cirque_center = fefcr.outerRadiusCenter;
    Point inner_cirque_center = fefcr.innerRadiusCenter;

    // set boundaries
    float roi_char_ub = fefcr.innerCirque_innerRadius * 1.18;		// 字符检测区域上边界
    float roi_char_lb = fefcr.innerCirque_innerRadius * 1.01;		// 字符检测区域下边界

    fefcr.character_roi_lb = roi_char_lb;
    fefcr.character_roi_ub = roi_char_ub;

    float min_theta = 0.96;								// 55°
    float max_theta = 2.18;								// 125°

    // get contours
    vector<vector<Point>> contours;
    Mat img_cart_bin;
    segmentImage(img_bin, img_cart_bin, contours, inner_cirque_center, roi_char_lb, roi_char_ub, min_theta, max_theta, 0);

    // get the corresponding parameters
    bool find_right_string = false;
    vector<Rect> boxes;
    if (contours.size() > 0)
    {
        getBoundingBoxes(contours, boxes);

        cout << "boxes size = " << boxes.size() << endl;
        if (boxes[0].x - fefcp.distance_to_border > 0 &&
            boxes[boxes.size() - 1].x + boxes[boxes.size() - 1].width + fefcp.distance_to_border < img_cart_bin.cols)
        {
            if (boxes.size() == 1)
            {
                find_right_string = true;
                std::cout << "boxes size = " << 1 << endl;

                fefcr.str_1_character.num      = 1;
                fefcr.str_1_character.width_lb = boxes[0].width - 10;
                fefcr.str_1_character.width_ub = boxes[0].width + 10;

                fefcr.str_1_character.height_lb = boxes[0].height - 10;
                fefcr.str_1_character.height_ub = boxes[0].height + 10;
            }
            else if (boxes.size() == 4)
            {
                find_right_string = true;
                std::cout << "boxes size = " << 4 << endl;

                int min_height = 0, max_height = 0;
                int min_width = 0, max_width = 0;
                int min_gap = 0, max_gap = 0;

                getCharacterParameters(boxes, min_width, max_width, min_height, max_height, min_gap, max_gap);

                fefcr.str_4_characters.num          = 4;
                fefcr.str_4_characters.width_lb     = min_width - 5;
                fefcr.str_4_characters.width_ub     = max_width + 5;
                fefcr.str_4_characters.height_lb    = min_height - 5;
                fefcr.str_4_characters.height_ub    = max_height + 5;
                fefcr.str_4_characters.small_gap_lb = min_gap - 2;
                fefcr.str_4_characters.small_gap_ub = max_gap + 2;
            }
            else if (boxes.size() == 10)
            {
                find_right_string = true;
                std::cout << "boxes size = " << 10 << endl;
                int min_height = 0, max_height = 0;
                int min_width = 0, max_width = 0;
                int min_gap = 0, max_gap = 0;

                getCharacterParameters(boxes, min_width, max_width, min_height, max_height, min_gap, max_gap);

                fefcr.str_10_characters.num          = 10;
                fefcr.str_10_characters.width_lb     = min_width - 5;
                fefcr.str_10_characters.width_ub     = max_width + 5;
                fefcr.str_10_characters.height_lb    = min_height - 5;
                fefcr.str_10_characters.height_ub    = max_height + 5;
                fefcr.str_10_characters.small_gap_lb = min_gap - 2;
                fefcr.str_10_characters.small_gap_ub = min_gap + 2;
                fefcr.str_10_characters.big_gap_lb   = max_gap - 5;
                fefcr.str_10_characters.big_gap_ub   = max_gap + 5;
            }
            else
            {
            }
        }
    }

    if (contours.size() > 0 && find_right_string)
    {
        Point pt_polar1;
        Point pt_cart;

        vector<Point> box_vector;
        find_right_string = false;
        for (size_t i = 0; i < boxes.size(); i++)
        {
            for (int j = boxes[i].y; j < boxes[i].y + boxes[i].height; j++)
            {
                pt_cart = Point(boxes[i].x, j + roi_char_lb);
                rhoTheta2xy(pt_cart, pt_polar1, inner_cirque_center, roi_char_ub, min_theta);
                box_vector.push_back(pt_polar1 + Point(fefcp.preSetArea.x, fefcp.preSetArea.y));

                pt_cart = Point(boxes[i].x + boxes[i].width, j + roi_char_lb);
                rhoTheta2xy(pt_cart, pt_polar1, inner_cirque_center, roi_char_ub, min_theta);
                box_vector.push_back(pt_polar1 + Point(fefcp.preSetArea.x, fefcp.preSetArea.y));
            }

            for (int k = boxes[i].x; k < boxes[i].x + boxes[i].width; k++)
            {
                pt_cart = Point(k, boxes[i].y + roi_char_lb);
                rhoTheta2xy(pt_cart, pt_polar1, inner_cirque_center, roi_char_ub, min_theta);
                box_vector.push_back(pt_polar1 + Point(fefcp.preSetArea.x, fefcp.preSetArea.y));

                pt_cart = Point(k, boxes[i].y +boxes[i].height + roi_char_lb);
                rhoTheta2xy(pt_cart, pt_polar1, inner_cirque_center, roi_char_ub, min_theta);
                box_vector.push_back(pt_polar1 + Point(fefcp.preSetArea.x, fefcp.preSetArea.y));
            }
        }
        fefcr.boxes_vector = box_vector;
    }
}


//! get min and max
void FrontEndFace::getMinandMax(int& value, int& minimum, int& maximum)
{
    if (value < minimum)
    {
        minimum = value;
    }

    if (value > maximum)
    {
        maximum = value;
    }
}

//! get bounding boxes
void FrontEndFace::getBoundingBoxes(vector<vector<Point>>& contours, vector<Rect>& boxes)
{
    if (contours.size() > 1)
    {
        for (size_t i = 0; i < contours.size(); i++)
        {
            boxes.push_back(boundingRect(Mat(contours[i])));
        }
        sort(boxes.begin(), boxes.end(), rect_x_sort);
    }
    else
    {
        boxes.push_back(boundingRect(Mat(contours[0])));
    }
}

//! get character parameters
void FrontEndFace::getCharacterParameters(vector<Rect>& boxes, int& min_width, int& max_width,
                            int& min_height, int& max_height, int& min_gap, int& max_gap)
{
    min_height = boxes[0].height;
    max_height = boxes[0].height;
    min_width  = boxes[0].width;
    max_width  = boxes[0].width;
    min_gap    = 10000;
    max_gap    = 0;

    for (size_t i = 1; i < boxes.size(); i++)
    {
        getMinandMax(boxes[i].height, min_height, max_height);
        getMinandMax(boxes[i].width, min_width, max_width);

        int temp = boxes[i].x - boxes[i - 1].x - boxes[i - 1].width;
        getMinandMax(temp, min_gap, max_gap);
    }
}


void FrontEndFace::rhoTheta2xy(cv::Point ptCart, cv::Point& ptPolar, cv::Point center, float radius, float min_theta)
{
    ptPolar.x = center.x + (double)ptCart.y * cos((double)ptCart.x / radius + PI + min_theta);
    ptPolar.y = center.y + (double)ptCart.y * sin((double)ptCart.x / radius + PI + min_theta);
}


void FrontEndFace::segmentImage(cv::Mat& img_bin, cv::Mat& img_cart_bin, vector<vector<Point>>& contours,
                                cv::Point& cirque_center, float& min_radius, float& max_radius,
                                float& min_theta, float& max_theta, int area_threshold)
{
    //! transform circular area into rectangle area
    polar2cart(img_bin, img_cart_bin, cirque_center, max_radius, min_theta, max_theta, min_radius);

    //! dilate the image
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(img_cart_bin, img_cart_bin, element, cv::Point(-1, -1), 1);

    //! remove the small connected components
    bwareaopen(img_cart_bin, 100);

    //! find contours
    findContours(img_cart_bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
}


// detect characters
bool FrontEndFace::detCharacters(FrontEndFaceDetParam2& fedp, FrontEndFaceDetResult2& fedr, int& ith_image)
{
    cv::Mat img_preset = fedp.imgSrc(fedp.fecp.preSetArea);

    cv::Mat img_bin;
    threshold(img_preset, img_bin, fedp.fecp.componentExistThreshold, 255, THRESH_BINARY_INV);

    Point inner_cirque_center;
    float inner_cirque_radius = 0;

    float angle_left  = fedp.fecp.angleLeft;
    float angle_right = fedp.fecp.angleRight;

    //! get center and radius
    vector<Point> edge_points;
    getEndFaceOuterRadiusPoints(img_bin, edge_points);
    circleLeastFit(edge_points, inner_cirque_center, inner_cirque_radius);

    float roi_char_ub = inner_cirque_radius * 1.158;		// 字符检测区域上边界
    float roi_char_lb = inner_cirque_radius * 1.01;		    // 字符检测区域下边界

    //! segment image
    vector<vector<Point>> contours;
    Mat img_cart_bin;

    segmentImage(img_bin, img_cart_bin, contours, inner_cirque_center, roi_char_lb, roi_char_ub, angle_right, angle_left, 0);

    //! case Judgement
    int case_n = 0;
    vector<Rect> boxes;
    caseJudgement(contours, boxes, img_cart_bin.cols, fedp.fecp.distance_to_border, case_n);


    if (case_n == 1)
    {
        int temp = boxes.size() / 2;
        int average = 0.0;

        if (boxes.size() % 2)
        {
            average = boxes[temp].x + boxes[temp].width / 2;
        }
        else
        {
            average = (boxes[temp - 1].x + boxes[temp - 1].width + boxes[temp].x) / 2;
        }
        ith_image += int((85 + average / ((2.18 - 0.96) * roi_char_ub) * 180 / CV_PI) / 6);
        return true;
    }
    else if (case_n == 2)
    {
        ith_image += int((boxes[0].x - 20) / ((2.18 - 0.96) * roi_char_ub) * 180 / CV_PI / 6);
        return false;
    }
    else if (case_n == 3)
    {
        ith_image += 60 / 6;
        return false;
    }
    else if (case_n == 4)
    {
        ith_image += 60 / 6;
        return false;
    }
}


void FrontEndFace::caseJudgement(vector<vector<Point>>& contours,
                                 vector<Rect>& boxes, int& upper_bound,
                                 int& distance_to_border,int& case_n)
{
    if (contours.size() > 0)
    {
        for (size_t i = 0; i < contours.size(); i++)
        {
            boxes.push_back(boundingRect(Mat(contours[i])));
        }
        sort(boxes.begin(), boxes.end(), rect_x_sort);

        if (contours.size() == 1 || contours.size() == 4 || contours.size() == 10)
        {
            if ((boxes[0].x - distance_to_border) > 0 && (boxes[boxes.size() - 1].x + boxes[boxes.size() - 1].width + distance_to_border) < upper_bound)
            {
                case_n = 1;
            }
            else if ((boxes[0].x - distance_to_border > 0) && (boxes[boxes.size() - 1].x + boxes[boxes.size() - 1].width + distance_to_border) > upper_bound)
            {
                case_n = 2;
            }
            else if ((boxes[0].x - distance_to_border < 0) && (boxes[boxes.size() - 1].x + boxes[boxes.size() - 1].width + distance_to_border) < upper_bound)
            {
                case_n = 3;
            }
            else
            {
                case_n = 4;
            }
        }
        else
        {
            if (boxes[0].x - distance_to_border > 0 && (boxes[boxes.size() - 1].x + boxes[boxes.size() - 1].width + distance_to_border) > upper_bound)
            {
                case_n = 2;
            }
            else if (boxes[0].x - distance_to_border < 0 && (boxes[boxes.size() - 1].x + boxes[boxes.size() - 1].width + distance_to_border) < upper_bound)
            {
                case_n = 3;
            }
            else
            {
                case_n = 4;
            }
        }
    }
    else
    {
        case_n = 4;
    }
}


