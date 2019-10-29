#pragma once
#include "DVInterface.h"
#include "DVEllipse.h"
#include "DVHoughCircle.h"
#include <omp.h>
#define PRINTLOG

using namespace Eigen;
enum DetectType//检测方法
{
    houghTransform,//hough变换
    curveFitting,//曲线拟合
    GridMinus,//梯度相减
};

enum cylinderDetPos //滚动体缺陷检测位置 cylinder detection position
{
    outerDiameterSurface,//外径面
    innerDiameterSurface,//内径面
    rollerendface,//滚子端面
    rollerlargeendface,//滚子大端面
    rollersmallendface,//滚子小端面
    outerChamfer,//外倒角 //Fillet 圆角
    innerChamfer,//内倒角
};

enum plotDefectAreaMethod
{
    CV_GLOBAL_AREA = 0,
    CV_LOCAL_AREA,
};



struct rolDetPosition //roller Detect Position
{
    cylinderDetPos rol_outerSurface; //滚子表面
    cylinderDetPos rol_smallEndFace;//滚子的小端面
    cylinderDetPos rol_largeEndFace;//滚子大端面
    cylinderDetPos rol_smallEndFaceChamfer;//滚子小端面倒角
    cylinderDetPos rol_largeEdnFaceChamfer;//滚子大端面倒角
};

struct ferruleDetPosition //套圈的检测位置
{
    cylinderDetPos fer_outerSurface;//套圈外径面
    cylinderDetPos fer_innerSurface;//套圈内径面
    cylinderDetPos fer_endface;//套圈的端面，目前左右端面没有分开处理，先这样，如果后期有改的需要在进行修改
    cylinderDetPos fer_innerChamfer;//套圈内侧倒角
    cylinderDetPos fer_outerChamfer;//套圈外侧倒角
};

struct completedBearing //成品轴承
{
    bool hasSealingRing;//是否含有密封圈
    cylinderDetPos bear_outerface;//成品轴承外径面
    cylinderDetPos bear_innerface;//成品轴承内径面
    cylinderDetPos bear_endface;//成品轴承端面
    cylinderDetPos bear_outerChamfer;//成品轴承外倒角
    cylinderDetPos bear_innerChamfer;//成品轴承内倒角

};

struct DetectPosition
{
    /*bool hasSealingRing;//是否含有密封圈*/
    DeviceType devicetype;
    cylinderDetPos cydetPos;
};


struct findPeakParam //查找峰值点的参数
{
    int minGap;//波峰之间的最小间距
    int minHeight;
};


struct cCParam //连通域的参数 connected components
{
    int STAT_AREA=0;//连通域的面积
    int STAT_WIDTH=0;//连通域的宽度
    int STAT_HEIGHT=0;//连通域的高度
    int STAT_HEIGHT_EDGE=0;//
    int STAT_HEIGHT_EDGE_down=0;
    int STAT_TOP=0;
};

class DVAlgorithm:DVInterface //DeepVision 算法类
{
public:
    DVAlgorithm();
    ~DVAlgorithm();
public:
    //辅助函数
    //y 降序
    static int cmpDescending(const pair<int, float>& x, const pair<int, float>&y);
    //y 升序
    static int cmpAscending(const pair<int, float>& x, const pair<int, float>&y);
    //查找轮廓点的时候的比较排序函数//根据选到的点的个数进行排序
    static bool contoursSort(const std::vector<Point>& pt1, const std::vector<Point>& pt2);
    //根据点的横向坐标进行排序
    static bool points_x_sort(const Point& pt1, const Point& pt2);
    //根据点的纵向坐标进行排序
    static bool points_y_sort(const Point& pt1, const Point& pt2);
    //根据rect的横坐标排序
    static bool rect_x_sort(const Rect& rect1,const Rect& rect2);

public:
    //圆转方
    /**
    * @name   polar2cart
    * @brief  圆转方
    * @param  [in] cv::Mat & mat_p
    * @param  [out] cv::Mat & mat_c
    * @param  [in] cv::Point2f center 输入待转换圆的圆心
    * @param  [in] float radius 输入待转换圆的半径
    * @return bool 如果转换成功返回true
    */
    //圆转方
    bool polar2cart(cv::Mat& mat_p, cv::Mat& mat_c, cv::Point2f center, float radius);
    //方转圆
    //自己写的方转圆 这里需要注意一个问题，就是mat_p需要指定返回图像的width,height
    //mat_p 需要在调用函数前指定大小
    bool cart2polar(cv::Mat& mat_c, cv::Mat& mat_p, cv::Point center);
    //圆转方 指定转换角度
    bool polar2cart(cv::Mat& mat_p,cv::Mat& mat_c,cv::Point2f center, float outer_radius,float thetaMin,float thetaMax,float rmin);
    //笛卡尔坐标系下的点集合转极坐标下的点  //方转圆 传入的半径为原转换的大圆的半径
    bool pointCart2polar(vector<Point> ptCart, vector<Point>& ptPolar, cv::Point center,float radius);
    //有一个偏移角度的情况
    bool pointCart2polar(vector<Point> ptCart,vector<Point>& ptPolar,cv::Point center,float radius,float thetaOffset);
    //笛卡尔坐标系下的点集合转极坐标下的点 //圆转方
    bool pointPolar2cart(vector<Point> ptPolar, vector<Point>& ptCart, cv::Point center, float radius);

    //笛卡尔坐标系下的点转极坐标下的点  //方转圆
    bool pointCart2polar(cv::Point ptCart, cv::Point& ptPolar, cv::Point center, float radius);
    //笛卡尔坐标系下的点转极坐标下的点 //圆转方
    bool pointPolar2cart(cv::Point ptPolar, cv::Point& ptCart, cv::Point center, float radius);
    //计算坐标y的平均值
    float calcAvg(vector<Point>& oriPts);
    //回归优化的方法
    //SSE(和方差)  该统计参数计算的是拟合数据和原始数据对应点的误差的平方和
    float calcSSE(vector<Point>& fitPts, vector<Point>& oriPts);
    //Sum of squares of the regression 即预测数据与原始数据均值之差的平方和
    float calcSSR(vector<Point>& prePts,float avg_y);
    //Total sum of squares，即原始数据和均值之差的平方和
    float calcSST(vector<Point>& oriPts, float avg_y);
    //R-square = ssr/sst   =(1-sse/sst);
    float calcRsquare(float ssr,float sst);


    /**
    * @name   circleLeastFit
    * @brief  用最小二乘法求出圆心及半径
    * @param  [in] const std::vector <cv::Point> points
    * @param  [out] cv::Point2f & center
    * @param  [out] float & radius
    * @return void
    */
    void circleLeastFit(const std::vector <cv::Point> points, cv::Point &center, float &radius);
    /**
    * @name   findPeaks
    * @brief  查找图像的峰值
    * @param  [in] const Mat Matdata
    * @param  [in] int minpeakdistance
    * @param  [in] int minpeakheight
    * @param  [in] std::map <int,float> & mapPeaks_max
    * @param  [in] vector<pair<int,float>> & Vector_map_Peaks_max
    * @return void
    */
    void findPeaks(const Mat Matdata, int minpeakdistance, int minpeakheight, std::map <int, float> &mapPeaks_max, vector<pair<int, float>>&Vector_map_Peaks_max);
    /**
    * @name   findPeaks
    * @brief  查找一维矩阵的峰值 注意这里输入的MatrixXf 与cv::Mat 输入的rows 和cols 取数据不一样，在eigen中是方法
    * @param  [in] Eigen::MatrixXf mat 输入的矩阵
    * @param  [in] int minpeakdistance 输入设定的最小峰值间的距离
    * @param  [in] int minpeakheight 输入设定的峰值的最小高度
    * @param  [out] std::map <int,float> & mapPeaks_max 结果输出的峰值map集合
    * @param  [out] vector<pair<int,float>> & Vector_map_Peaks_max 结果输出 对上面map进行排序后的结果
    * @return void
    */
    void findPeaks(Eigen::MatrixXf mat, int minpeakdistance, int minpeakheight, std::map <int, float> &mapPeaks_max, vector<pair<int, float>>&Vector_map_Peaks_max);
    //对峰值map 进行排序
    void sortMapByValue(map<int, float>& tMap, vector<pair<int, float>>& tVector);
    //查找一维矩阵中的最大项的位置
    void findMax(Eigen::MatrixXf mat, std::map<int, float>& mapPeaks_max);
    //得到两个点之间的距离
    float getDistance(cv::Point pointA, cv::Point pointB);
    /**
    * @name   GetGrayAvgStdDev
    * @brief  计算图像平均灰度与灰度方差的程序
    * @param  [in] cv::Mat & src
    * @param  [out] float & avg 计算出的图像的平均灰度
    * @param  [out] float & stddev 计算出的图像的灰度方差
    * @return void
    */
    void GetGrayAvgStdDev(cv::Mat& src, float& avg, float &stddev);
    /**
    * @name   画出图像横向上的投影特征
    * @brief  测试采用投影的方法求检测目标区域范围
    * @param  [in] cv::Mat imgsrc
    * @return void
    */
    void drawX_img(cv::Mat& imgsrc);
    //画出图像纵向上的投影特征
    void drawY_img(cv::Mat& imgsrc);
    //多项式曲线拟合
    bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);

    /**
    * @name   showImg
    * @brief  以原图图像大小的1/2大小显示
    * @param  [in] string imgname 显示的名字
    * @param  [in] cv::Mat imgsrc 要显示的原始图像
    * @return void
    */
    void showImg(string imgname, cv::Mat& imgsrc);
    //两幅图像相减,第一幅图像相减前的原图像，第二幅图像为卷积运算滤波后的图像
    void minusImg(cv::Mat imgroi, float a1, cv::Mat imgcov, float a2, cv::Mat& imgout);
    //得到图像的垂直和水平投影
    void getImgHVHist(Mat &src, Mat &Hdist, Mat &Vdist);
    //得到图像的水平投影，求水平最小平均灰度及对应的行号,采用opencvReduce方法，不太行计算有误（还在找问题）
    void getImgHmap_min(cv::Mat& src, map<int, int>& cartpos_Threshold);
    void getImgHmap_min_filter(cv::Mat& src, map<int, int>& cartpos_Threshold);

    //得到内部圆的边界点的集合//采用点扫描的方式
    void getBounderInnerCoutours(cv::Mat& imgbin, vector<Point>& points,float innerRaidusEstimate,int OFFSET);
    Mat polyfit(vector<Point>& in_point, int n);
    //传入二值化后的图像，通过横向扫描获取白色点区域的最大最小值，目前是所有点都拿取，
    //如果后期时间效率有待提高的话，可以筛选其中几个点进行处理
    void getWhiteCircleContours(cv::Mat& imgbin, vector<cv::Point>& ptcontours);
    //得到3次曲线拟合的点的集合
    void getploy3Contours(vector<Point>& srcContours,vector<Point>& dstContours);

    //找到一组点的最大点和最小的点的坐标
    void getMinMaxYPoint(vector<Point>& pointsSrc,cv::Point& ptMax,cv::Point& ptMin);
    //大尺度滤波函数 //
    void blurImg_gash(cv::Mat& imgsrc,Eigen::MatrixXf& filter,cv::Mat& imgout);
    //TODO：这里先不加后面连通域参数了
    void bwareaopen(cv::Mat& imgsrc, cCParam ccp);
    void bwareaopen(Mat& src,int areaThreshold);
    void bwareaopen(Mat& src,cCParam ccp,vector<cv::Rect>& results);
    void bwareaopen(Mat& src, Mat &dst, cCParam ccp,vector<cv::Rect>& results);
    void bwareaclose(cv::Mat& src,cCParam ccp);
public:
    //取点集合的三个点做平均运算
    void pointblur(vector<cv::Point> pt, vector<cv::Point>& ptResult, int step);
    //virtual void getRoiRect(cv::Mat imggray);
    int createXML(const char* xmlPath,const char* xmlRootName);
    void getDistanceByHist(cv::Mat& imgtemplate,cv::Mat& imgtest,float& correlationDistance,float& bDistance,float& emdDistance);
    void DVsobel(cv::Mat imgsrc,cv::Mat& imgout);
    void DVrobert(cv::Mat imgsrc,cv::Mat& imgout);
    void plotDefectArea(cv::Mat& src, cv::Mat polarbin,
                        plotDefectAreaMethod pdam, int roffset,
                        int toffset, cv::Point Center, int rmax);

    void getRectsBorderPoints(vector<Rect> rects,vector<Point>& pts);
    void points2HoleCartImg(vector<Point> defPointsOnDetImg,int imgCartOffset,vector<Point>& defPointsOnHoleCart);
    void points2HoleCartImg(vector<Point> defPointsOnDetImg,int imgCartOffset,int imgCartXShiftOffset,int imgCols,vector<Point>& defPointsOnHoleCart);
    void pointsHoleCart2ImgSrc(vector<Point> defPointsOnHoleCart,cv::Point center,int outerRadius,cv::Rect presetArea,vector<Point>& defPointsOnImgSrc);

    void drawDefPointsOnImgResult(cv::Mat& imgResult,vector<Point> defPoints);
};


