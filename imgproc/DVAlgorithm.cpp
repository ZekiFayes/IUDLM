#include "DVAlgorithm.h"



DVAlgorithm::DVAlgorithm()
{
}


DVAlgorithm::~DVAlgorithm()
{
}

bool DVAlgorithm::contoursSort(const std::vector<Point>& pt1,const std::vector<Point>& pt2)
{
    return pt1.size() > pt2.size();
}

bool DVAlgorithm::points_x_sort(const Point& pt1, const Point& pt2)//点X从小到大排列
{
    return pt1.x < pt2.x;
}
bool DVAlgorithm::points_y_sort(const Point& pt1, const Point& pt2)
{
    return pt1.y > pt2.y;
}

bool DVAlgorithm::rect_x_sort(const Rect& rect1,const Rect& rect2)
{
    return rect1.x<rect2.x;
}



//辅助函数
//降序
int DVAlgorithm::cmpDescending(const pair<int, float>& x, const pair<int, float>&y)
{
    return  x.second < y.second;
}
//升序
int DVAlgorithm::cmpAscending(const pair<int, float>& x, const pair<int, float>&y)
{
    return  x.second > y.second;
}


//完成map到vector的转化并且完成排序
void DVAlgorithm::sortMapByValue(map<int, float>& tMap, vector<pair<int, float>>& tVector)
{
    for (map<int, float>::iterator curr = tMap.begin(); curr != tMap.end(); curr++)
    {
        tVector.push_back(make_pair(curr->first, curr->second));
    }
    sort(tVector.begin(), tVector.end(), cmpAscending);
}

//圆转方
bool DVAlgorithm::polar2cart(cv::Mat& mat_p, cv::Mat& mat_c, cv::Point2f center, float radius)
{
    if(!mat_p.data)
    {
        dzlog_error("polar2cart img mat_p has no data ...");
        return false;
    }
    int rows_c = (int)std::ceil(radius);//这里通过传入的半径范围确定转换成的
    int cols_c = (int)std::ceil(2.0 * CV_PI * radius);

    dzlog_debug("极坐标图的宽度 == %d",cols_c);
    dzlog_debug("极坐标图的高度 == %d",rows_c);

//    std::cout << "利用极坐标转换表示的方图 polarImg.cols_c==" << cols_c << endl;
    mat_c = cv::Mat::zeros(rows_c, cols_c, CV_8UC1);

    //int polar_d = mat_p.cols;
    float polar_r = radius;

    float delta_r = polar_r / rows_c; //1
    float delta_t = 2.0*CV_PI / cols_c;  //1/r

    float center_polar_x = center.x;
    float center_polar_y = center.y;

    for (int i = 0; i < cols_c-1; i++)
    {
        float theta_p = i * delta_t;
        float sin_theta = std::sin(theta_p);
        float cos_theta = std::cos(theta_p);
        for (int j = 0; j < rows_c-1; j++)
        {
            float temp_r = j * delta_r;
            int polar_x = (int)(center_polar_x + temp_r * cos_theta);
            int polar_y = (int)(center_polar_y + temp_r * sin_theta);
            if(polar_x>=0 && polar_x<mat_p.cols && polar_y>=0 && polar_y<mat_p.rows)
            {
                mat_c.ptr<uchar>(j)[i] = mat_p.ptr<uchar>(polar_y)[polar_x];
            }
        }
    }
    return true;
}


//圆转方
bool DVAlgorithm::polar2cart(cv::Mat& mat_p,cv::Mat& mat_c,cv::Point2f center, float outer_radius,float thetaMin,float thetaMax,float rmin)
{
    if(!mat_p.data)
    {
        dzlog_error("polar2cart img mat_P has no data ...");
        return false;
    }
    int rows_c = (int)std::ceil(outer_radius-rmin);//这里通过传入的半径范围确定转换成的
    int cols_c = (int)std::ceil((thetaMax-thetaMin) * outer_radius);
    dzlog_debug("极坐标图的宽度 == %d",cols_c);
    dzlog_debug("极坐标图的高度 == %d",rows_c);

//    std::cout << "利用极坐标转换表示的方图 polarImg.cols_c==" << cols_c << endl;
    mat_c = cv::Mat::zeros(rows_c, cols_c, CV_8UC1);

    float rstep=1.0;
    float thetaStep=(thetaMax-thetaMin)/cols_c;

    float center_polar_x = center.x;
    float center_polar_y = center.y;

    for (int i = 0; i < cols_c-1; i++)
    {
        float theta_p =thetaMin + i * thetaStep+PI;//因为坐标系翻转
        float sin_theta = std::sin(theta_p);
        float cos_theta = std::cos(theta_p);

        for (int j = 0; j < rows_c-1; j++)
        {
            float temp_r =rmin + j * rstep;

            int polar_x = (int)(center_polar_x+temp_r * cos_theta);
            int polar_y = (int)(center_polar_y + temp_r * sin_theta);
            if(polar_x>0 && polar_x<mat_p.cols-1 && polar_y>0 && polar_y<mat_p.rows-1){
                mat_c.ptr<uchar>(j)[i] = mat_p.ptr<uchar>(polar_y)[polar_x];
            }
        }
    }
    return true;
}


/************************************************************************/
/* 根据点的集合得到圆的圆心及半径                                                                  */
/************************************************************************/
void DVAlgorithm::circleLeastFit(const std::vector <cv::Point> points, cv::Point &center, float &radius)
{
    dzlog_debug("get start circleLeastFit");
    radius = 0.0f;
    float sum_x = 0.0f, sum_y = 0.0f;
    float sum_x2 = 0.0f, sum_y2 = 0.0f;
    float sum_x3 = 0.0f, sum_y3 = 0.0f;
    float sum_xy = 0.0f, sum_x1y2 = 0.0f, sum_x2y1 = 0.0f;
    int N = (int)points.size();
    for (int i = 0; i < N; i++)
    {
        float x = points[i].x;
        float y = points[i].y;
        float x2 = x*x;
        float y2 = y*y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x2;
        sum_y2 += y2;
        sum_x3 += x2*x;
        sum_y3 += y2*y;
        sum_xy += x*y;
        sum_x1y2 += x*y2;
        sum_x2y1 += x2*y;
    }
    float C, D, E, G, H;
    float a, b, c;
    C = N*sum_x2 - sum_x*sum_x;
    D = N*sum_xy - sum_x*sum_y;
    E = N*sum_x3 + N*sum_x1y2 - (sum_x2 + sum_y2)*sum_x;
    G = N*sum_y2 - sum_y*sum_y;
    H = N*sum_x2y1 + N*sum_y3 - (sum_x2 + sum_y2)*sum_y;
    a = (H*D - E*G) / (C*G - D*D);
    b = (H*C - E*D) / (D*D - G*C);
    c = -(a*sum_x + b*sum_y + sum_x2 + sum_y2) / N;
    center.x = a / (-2);
    center.y = b / (-2);
    radius = sqrt(a*a + b*b - 4 * c) / 2;
}


/*
输入参数
Matdata: 输入的数据矩阵
minpeakdistance 设定两峰值间的最小间隔数
minpeakheight  设定峰值的最小高度
输出函数:
mapPeaks_max 按照key值(灰度级0-255)排序的
Vector_map_Peaks_max 按照value值(各灰度级的像素值)排序的
备注:此函数只完成了波峰的图提取,波谷自行完成
*/
void DVAlgorithm::findPeaks(const Mat Matdata, int minpeakdistance, int minpeakheight, std::map <int, float> &mapPeaks_max, vector<pair<int, float>>&Vector_map_Peaks_max)
{
    int row = Matdata.rows - 1;
    int col = Matdata.cols;
    vector<int> sign;
    vector<float> markdata;
    vector<float> markdata1;
    //minMaxLoc寻找矩阵(一维数组当做向量,用Mat定义)中最小值和最大值的位置
    //类型转换注意:data为unchar*
    float* pMatdata = (float*)Matdata.data;
    for (int i = 1; i < col; i++)
    {
        /*相邻做差:
        *小于0,赋值-1
        *大于0,赋值1
        *等于0,赋值0
        */
        markdata1.push_back(*(pMatdata + i - 1));
        int diff = *(pMatdata + i) - *(pMatdata + i - 1);
        if (diff > 0)
        {
            sign.push_back(1);
        }
        else if (diff < 0)
        {
            sign.push_back(-1);
        }
        else
        {
            sign.push_back(0);
        }
    }
    //再对sign相邻位做差
    //保存极大值和极小值的位置
    vector<int> indMax;
    vector<int> indMin;
    std::map <int, float> mapPeaks_min;
    for (int j = 1; j < sign.size(); j++)
    {
        int diff = sign[j] - sign[j - 1];
        if (diff < 0)
        {
            indMax.push_back(j);
            //根据峰值最小高度进行筛选
            if (*(pMatdata + indMax[indMax.size() - 1]) > minpeakheight) {
                mapPeaks_max.insert(pair<int, float>(j, *(pMatdata + indMax[indMax.size() - 1])));
            }

        }
        else if (diff>0)
        {
            indMin.push_back(j);
            mapPeaks_min.insert(pair<int, float>(j, *(pMatdata + indMin[indMin.size() - 1])));

        }
    }

    if (mapPeaks_max.size() >= 2) {

        //找到像素值最大的对应的容器指针
        std::map<int, float>::iterator iter_high_max1_pos = std::max_element(std::begin(mapPeaks_max), std::end(mapPeaks_max), cmpDescending);
        int high_max1_pos = iter_high_max1_pos->first;//最高点的位置
        map<int, float>::iterator  iter;
        map<int, float>::iterator  temp;
        //以位置为序
        //map<int, float>::iterator  iter_high_max1_pos;
        map<int, float>::iterator  iter_high_max1_pos_temp1;
        map<int, float>::iterator  iter_high_max1_pos_temp2;
        //知道峰值最高的对应容器指针
        //我们以最高峰值处进行作业搜索 所以在此声明两个局部变量保存该指针
        iter_high_max1_pos_temp1 = iter_high_max1_pos;
        iter_high_max1_pos_temp2 = iter_high_max1_pos;

        //rate为宽度,以iter_high_max1_pos为起点,根据minpeakdistance的距离倍数搜索
        //minpeakdistance宽度内取一个最大的峰值max_value_temp
        int rate = 1;
        float max_value_temp = 0;
        map<int, float>::iterator max_value_temp_iter;
        //每个minpeakdistance内的第一个指针防止被删除
        bool is_first_max = true;
        //挨着最大峰值的
        if (mapPeaks_max.size() >= 2) {
            if (iter_high_max1_pos_temp1 != mapPeaks_max.begin()) {

                for (iter = --iter_high_max1_pos_temp1; iter != mapPeaks_max.begin(); iter--) {
                    //小于minpeakdistance
                    if (labs((*iter).first - high_max1_pos) < (rate)*minpeakdistance) {
                        temp = iter;
                        iter++;
                        mapPeaks_max.erase(temp);
                    }
                    if (rate*minpeakdistance <= labs((*iter).first - high_max1_pos) && labs((*iter).first - high_max1_pos) <= (rate + 1)*minpeakdistance) {
                        if ((*iter).second > max_value_temp) {
                            max_value_temp = (*iter).second;
                            if (is_first_max) {
                                max_value_temp_iter = iter;
                                is_first_max = false;
                            }
                            else {
                                //删除原先最大的
                                mapPeaks_max.erase(max_value_temp_iter);
                                temp = iter;
                                iter++;
                                max_value_temp_iter = temp;
                            }
                        }
                        else {
                            is_first_max = true;
                            max_value_temp = 0;
                            temp = iter;
                            iter++;
                            mapPeaks_max.erase(temp);
                        }
                    }
                        //大于(rate + 1)*minpeakdistance
                    else if (labs((*iter).first - high_max1_pos)>(rate + 1)*minpeakdistance) {

                        rate++;
                        iter++;
                    }
                }
            }
            is_first_max = true;
            if (iter_high_max1_pos_temp2 != mapPeaks_max.end() && mapPeaks_max.size() >= 2) {

                for (iter = ++iter_high_max1_pos_temp2; iter != mapPeaks_max.end(); iter++) {
                    //TRACE("2Vector:%d \n", debug++);
                    //小于minpeakdistance
                    if (labs((*iter).first - high_max1_pos) < (rate)*minpeakdistance) {
                        temp = iter;
                        iter--;
                        mapPeaks_max.erase(temp);
                    }
                    if (rate*minpeakdistance <= labs((*iter).first - high_max1_pos) && labs((*iter).first - high_max1_pos) <= (rate + 1)*minpeakdistance) {

                        if ((*iter).second > max_value_temp) {
                            max_value_temp = (*iter).second;
                            if (is_first_max) {
                                max_value_temp_iter = iter;
                                is_first_max = false;
                            }
                            else {
                                //删除原来最大的
                                mapPeaks_max.erase(max_value_temp_iter);

                                temp = iter;
                                iter--;
                                max_value_temp_iter = temp;
                            }
                        }

                    }
                        //大于(rate + 1)*minpeakdistance
                    else if (labs((*iter).first - high_max1_pos)>(rate + 1)*minpeakdistance) {
                        //初始化
                        is_first_max = true;
                        max_value_temp = 0;
                        rate++;
                        iter--;

                    }
                }
            }
            //因为容器指针的begin()指针在上面的循环中未进行判断,所以这里单独处理
            //end()指的是最后一个元素的下一个位置所以不需要淡定进行判断
            map<int, float>::iterator max_value_temp_iter_more;
            if (mapPeaks_max.size() >= 2) {
                max_value_temp_iter = mapPeaks_max.begin();
                max_value_temp_iter_more = max_value_temp_iter++;
                if (labs(max_value_temp_iter->first - max_value_temp_iter_more->first) < minpeakdistance) {
                    if (max_value_temp_iter->second > max_value_temp_iter_more->second) {
                        mapPeaks_max.erase(max_value_temp_iter_more);
                    }
                    else {
                        mapPeaks_max.erase(max_value_temp_iter);
                    }

                }
            }
        }
        Vector_map_Peaks_max.clear();
        //二维向量排序(根据函数选择以key或者value排序)
        sortMapByValue(mapPeaks_max, Vector_map_Peaks_max);
    }
}


void DVAlgorithm::findPeaks(Eigen::MatrixXf mat, int minpeakdistance, int minpeakheight, std::map <int, float> &mapPeaks_max, vector<pair<int, float>>&Vector_map_Peaks_max)
{
    int row = mat.rows();
    int col = mat.cols();
    vector<int> sign;
    //minMaxLoc寻找矩阵(一维数组当做向量,用Mat定义)中最小值和最大值的位置
    //类型转换注意:data为unchar*
    float* pMatdata = (float*)mat.data();
    if (row > 1 && col == 1)
    {
        for (int i = 1; i < row; i++)
        {
            /*相邻做差:
       *小于0,赋值-1
       *大于0,赋值1
       *等于0,赋值0
       */
            //markdata1.push_back(*(pMatdata + i - 1));
            int diff = *(pMatdata + i) - *(pMatdata + i - 1);
            if (diff > 0)
            {
                sign.push_back(1);
            }
            else if (diff < -3)//FIXME
            {
                sign.push_back(-1);
            }
            else
            {
                sign.push_back(0);
            }
        }
    }
    else if (col > 1 && row == 1)
    {
        for (int i = 1; i < col; i++)
        {
            /*相邻做差:
       *小于0,赋值-1
       *大于0,赋值1
       *等于0,赋值0
       */
            //markdata1.push_back(*(pMatdata + i - 1));
            int diff = *(pMatdata + i) - *(pMatdata + i - 1);
            if (diff > 0)
            {
                sign.push_back(1);
            }
            else if (diff < 0)
            {
                sign.push_back(-1);
            }
            else
            {
                sign.push_back(0);
            }
        }
    }

    //再对sign相邻位做差
    //保存极大值和极小值的位置
    vector<int> indMax;
    vector<int> indMin;
    std::map <int, float> mapPeaks_min;
    for (int j = 1; j < sign.size(); j++)
    {
        int diff = sign[j] - sign[j - 1];
        if (diff < 0)
        {
            indMax.push_back(j);
            //根据峰值最小高度进行筛选
            if (*(pMatdata + indMax[indMax.size() - 1]) > minpeakheight) {
                mapPeaks_max.insert(pair<int, float>(j, *(pMatdata + indMax[indMax.size() - 1])));
            }
        }
        else if (diff>0)
        {
            indMin.push_back(j);
            mapPeaks_min.insert(pair<int, float>(j, *(pMatdata + indMin[indMin.size() - 1])));
        }
    }

    if (mapPeaks_max.size() >= 2) {

        std::map<int, float>::iterator iter_high_max1_pos = std::max_element(std::begin(mapPeaks_max), std::end(mapPeaks_max), cmpDescending);
        int high_max1_pos = iter_high_max1_pos->first;
        map<int, float>::iterator  iter;
        map<int, float>::iterator  temp;

        //map<int, float>::iterator  iter_high_max1_pos;
        map<int, float>::iterator  iter_high_max1_pos_temp1;
        map<int, float>::iterator  iter_high_max1_pos_temp2;

        iter_high_max1_pos_temp1 = iter_high_max1_pos;
        iter_high_max1_pos_temp2 = iter_high_max1_pos;


        int rate = 1;
        float max_value_temp = 0;
        map<int, float>::iterator max_value_temp_iter;

        bool is_first_max = true;

        if (mapPeaks_max.size() >= 2) {
            if (iter_high_max1_pos_temp1 != mapPeaks_max.begin()) {

                for (iter = --iter_high_max1_pos_temp1; iter != mapPeaks_max.begin(); iter--) {
                    //小于minpeakdistance
                    if (labs((*iter).first - high_max1_pos) < (rate)*minpeakdistance) {
                        temp = iter;
                        iter++;
                        mapPeaks_max.erase(temp);
                    }
                    int distancetemp = labs((*iter).first - high_max1_pos);
                    if (rate*minpeakdistance <= labs((*iter).first - high_max1_pos) && labs((*iter).first - high_max1_pos) <= (rate + 1)*minpeakdistance) {
                        if ((*iter).second > max_value_temp) {
                            max_value_temp = (*iter).second;
                            if (is_first_max) {
                                max_value_temp_iter = iter;
                                is_first_max = false;
                            }
                            else {

                                mapPeaks_max.erase(max_value_temp_iter);
                                temp = iter;
                                iter++;
                                max_value_temp_iter = temp;
                            }
                        }
                        else {
                            is_first_max = true;
                            max_value_temp = 0;
                            temp = iter;
                            iter++;
                            mapPeaks_max.erase(temp);
                        }
                    }

                    else if (labs((*iter).first - high_max1_pos) > (rate + 1)*minpeakdistance) {
                        rate++;
                        iter++;
                    }
                }
            }
            is_first_max = true;
            if (iter_high_max1_pos_temp2 != mapPeaks_max.end() && mapPeaks_max.size() >= 2) {
                for (iter = ++iter_high_max1_pos_temp2; iter != mapPeaks_max.end(); iter++) {
                    //TRACE("2Vector:%d \n", debug++);

                    if (labs((*iter).first - high_max1_pos) < (rate)*minpeakdistance) {
                        temp = iter;
                        iter--;
                        mapPeaks_max.erase(temp);
                    }
                    if (rate*minpeakdistance <= labs((*iter).first - high_max1_pos) && labs((*iter).first - high_max1_pos) <= (rate + 1)*minpeakdistance) {

                        if ((*iter).second > max_value_temp) {
                            max_value_temp = (*iter).second;
                            if (is_first_max) {
                                max_value_temp_iter = iter;
                                is_first_max = false;
                            }
                            else {

                                mapPeaks_max.erase(max_value_temp_iter);

                                temp = iter;
                                iter--;
                                max_value_temp_iter = temp;
                            }
                        }

                    }

                    else if (labs((*iter).first - high_max1_pos) > (rate + 1)*minpeakdistance) {

                        is_first_max = true;
                        max_value_temp = 0;
                        rate++;
                        iter--;

                    }
                }
            }

            map<int, float>::iterator max_value_temp_iter_more;
            if (mapPeaks_max.size() >= 2) {
                max_value_temp_iter = mapPeaks_max.begin();
                max_value_temp_iter_more = max_value_temp_iter++;
                if (labs(max_value_temp_iter->first - max_value_temp_iter_more->first) < minpeakdistance) {
                    if (max_value_temp_iter->second > max_value_temp_iter_more->second) {
                        mapPeaks_max.erase(max_value_temp_iter_more);
                    }
                    else {
                        mapPeaks_max.erase(max_value_temp_iter);
                    }

                }
            }
        }
        Vector_map_Peaks_max.clear();

        sortMapByValue(mapPeaks_max, Vector_map_Peaks_max);
    }
}



//查找一维矩阵 向量的峰值位置及数值 二分法
void DVAlgorithm::findMax(Eigen::MatrixXf mat, std::map<int, float>& mapPeaks_max)
{
    if (mat.cols() > 1 && mat.rows() == 1)
    {
        if (mat.cols() < 3) {
            return;
        }
        int start = 1;
        int end = mat.cols() - 2;
        while (start + 1 < end) {
            int middle = start + (end - start) / 2;
            if (mat(middle) < mat(middle - 1)) {
                end = middle;
            }
            else if (mat(middle) < mat(middle + 1)) {
                start = middle;
            }
            else {
                start = middle;
            }
        }
        if (mat(start) > mat(end)) {
            mapPeaks_max.insert(std::pair<int, float>(start, mat(start)));
            return;
        }
        mapPeaks_max.insert(pair<int, float>(end, mat(end)));
    }
    else if (mat.rows() > 1 && mat.cols() == 1) {
        if (mat.rows() < 3) {
            return;
        }
        int start = 1;
        int end = mat.rows() - 2;
        while (start + 1 < end) {
            int middle = start + (end - start) / 2;
            if (mat(middle) < mat(middle - 1)) {
                end = middle;
            }
            else if (mat(middle) < mat(middle + 1)) {
                start = middle;
            }
            else {
                start = middle;
            }
        }
        if (mat(start) > mat(end)) {
            mapPeaks_max.insert(pair<int, float>(start, mat(start)));
            return;
        }
        mapPeaks_max.insert(pair<int, float>(end, mat(end)));
    }
}

float DVAlgorithm::getDistance(cv::Point pointO, cv::Point pointA)
{
    float distance;
    distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
    distance = sqrtf(distance);
    return distance;
}

void DVAlgorithm::GetGrayAvgStdDev(cv::Mat& src, float& avg, float &stddev)
{
    cv::Mat img;
    if (src.channels() == 3)
        cv::cvtColor(src, img, CV_BGR2GRAY);
    else
        img = src;
    cv::mean(src);
    cv::Mat mean;
    cv::Mat stdDev;
    cv::meanStdDev(img, mean, stdDev);

    avg = mean.ptr<double>(0)[0];
    stddev = stdDev.ptr<double>(0)[0];
}

//灰度横向投影
void DVAlgorithm::drawX_img(cv::Mat& imggray)
{
    int height = imggray.rows;//高度
    int width = imggray.cols;//宽度
    MatrixXf imgsrc_matrix(height, width);
    //Eigen::Matrix<int, Dynamic, Dynamic> imgsrc_matrix;
    cv2eigen(imggray, imgsrc_matrix);
    Eigen::MatrixXf filter_x = MatrixXf::Ones(width, 1);
    Eigen::MatrixXf row_sum_matrix(height, 1);
    row_sum_matrix = imgsrc_matrix*filter_x;
    row_sum_matrix = row_sum_matrix / width;
    //std::cout << "横向投影  "<< row_sum_matrix << endl;
    std::vector<Point> x_points;
    for (int i = 0; i < height; i++)
    {
        x_points.push_back(cv::Point(i, row_sum_matrix(i)));
    }
    //创建用于绘制的深蓝色背景图像
    cv::Mat x_image = cv::Mat::zeros(255, height, CV_8UC3);
    x_image.setTo(cv::Scalar(100, 0, 0));
    //将拟合点绘制到空白图上
    for (int i = 0; i < x_points.size(); i++)
    {
        cv::circle(x_image, x_points[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    //绘制折线
    cv::polylines(x_image, x_points, false, cv::Scalar(0, 255, 0), 1, 8, 0);
    cv::Mat A;
    polynomial_curve_fit(x_points, 3, A);
    std::cout << "A = " << A << std::endl;
    std::vector<cv::Point> points_fitted;
    for (int x = 0; x < height; x++)
    {
        float y = A.at<float>(0, 0) + A.at<float>(1, 0) * x +
                  A.at<float>(2, 0)*std::pow(x, 2) + A.at<float>(3, 0)*std::pow(x, 3);
        points_fitted.push_back(cv::Point(x, y));
    }
    cv::polylines(x_image, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
    cv::imshow("x_image", x_image);
    cv::waitKey(0);
}

//灰度纵向投影
void DVAlgorithm::drawY_img(cv::Mat& imggray)
{
    int height = imggray.rows;//高度
    int width = imggray.cols;//宽度
    MatrixXf imgsrc_matrix(height, width);
    cv2eigen(imggray, imgsrc_matrix);
    Eigen::MatrixXf filter_y = MatrixXf::Ones(1, height);
    Eigen::MatrixXf col_sum_matrix(1, width);
    col_sum_matrix = filter_y*imgsrc_matrix;
    col_sum_matrix = col_sum_matrix / height;
    //std::cout << "纵向投影   " << col_sum_matrix << endl;
    std::vector<Point> y_points;
    for (int i = 0; i < width; i++)
    {
        y_points.push_back(cv::Point(i, col_sum_matrix(i)));
    }
    //创建用于绘制的深蓝色背景图像
    cv::Mat y_image = cv::Mat::zeros(255, width, CV_8UC3);
    y_image.setTo(cv::Scalar(100, 0, 0));
    //将拟合点绘制到空白图上
    for (int i = 0; i < y_points.size(); i++)
    {
        cv::circle(y_image, y_points[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    //绘制折线
    cv::polylines(y_image, y_points, false, cv::Scalar(0, 255, 0), 1, 8, 0);
    cv::Mat A;
    polynomial_curve_fit(y_points, 3, A);
    std::cout << "A = " << A << std::endl;
    std::vector<cv::Point> points_fitted;
    for (int x = 0; x < height; x++)
    {
        float y = A.at<float>(0, 0) + A.at<float>(1, 0) * x +
                  A.at<float>(2, 0)*std::pow(x, 2) + A.at<float>(3, 0)*std::pow(x, 3);
        points_fitted.push_back(cv::Point(x, y));
    }
    cv::polylines(y_image, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
    cv::imshow("y_image", y_image);
    cv::waitKey(0);
}

bool DVAlgorithm::polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = (int)key_point.size();

    //构造矩阵X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
//                std::cout << "key_point[k].x == " << key_point[k].x << endl;
                X.at<float>(i, j) = X.at<float>(i, j) +
                                    std::pow(key_point[k].x, i + j);
            }
        }
    }

    //构造矩阵Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<float>(i, 0) = Y.at<float>(i, 0) +
                                std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }
    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //求解矩阵A
    cv::solve(X, Y, A, cv::DECOMP_LU);
    return true;
}

void DVAlgorithm::showImg(string imgname, cv::Mat& imgsrc)//FIXME:这里有一个大坑,目前已经解决了
{
//    return;
    cv::Mat imgdist;
    cv::resize(imgsrc, imgdist, imgdist.size(), 0.5, 0.5);
    cv::imshow(imgname, imgdist);
    cv::waitKey(0);
}

//TODO:这里可以加速
void DVAlgorithm::minusImg(cv::Mat imgsrc, float a1, cv::Mat imgcov, float a2, cv::Mat& imgout)
{
    dzlog_debug("get start minusImg ...");
    cv::Mat result1 = imgsrc.clone();
    cv::Mat result2 = imgcov.clone();
    int rownum = result1.rows;
    int colnum = result1.cols;
    for (int i = 0; i < rownum; i++)
    {
        for (int j = 0; j < colnum; j++)
        {
            //result1.at<Vec2b>(i, j)[0] = toZero(imgsrc.at<Vec3b>(i, j)[0] - imgcov.at<Vec3b>(i, j)[0]);
            result1.ptr(i)[j] = abs(imgcov.ptr(i)[j] - imgsrc.ptr(i)[j]);
        }
    }
    addWeighted(imgcov, 1, imgsrc, -1, 0, result2);
    imgout = result1;
}


void DVAlgorithm::getImgHVHist(Mat &src, Mat &Hdist, Mat &Vdist)
{
    dzlog_debug("get start getImgHVHist ...");
    if(!src.data)
    {
        dzlog_debug("err getImgHVhist imgsrc is empty ...");
    }
    //投影法
    Mat image = src.clone();
    int width = src.cols;
    int height = src.rows;
    dzlog_debug("img.width == %d",width);
    dzlog_debug("img.height == %d",height);
    Hdist = Mat::zeros(height, 1, CV_32FC1);//CV_32FC1,可根据实际取值范围修改
    Vdist = Mat::zeros(1, width, CV_32FC1);
    //CV_REDUCE_SUM-输出是矩阵的所有行/列的和 //CV_REDUCE_AVG-输出的是矩阵所有行/列的平均向量
    //CV_REDUCE_MAX-输出是矩阵的所有行/列的最大值 //CV_REDUCE_MIN-出书是矩阵的所有行/列的最小值
    //沿水平方向投影,得到height行,1列的矩阵
    reduce(image, Hdist, 1, CV_REDUCE_SUM, CV_32FC1);//rtype要与创建dst的type一致

    //沿垂直方向投影,得到1行,width列的矩阵
    reduce(image, Vdist, 0, CV_REDUCE_SUM, CV_32FC1);//rtype要与创建dst的type一致
}

bool DVAlgorithm::cart2polar(cv::Mat& mat_c, cv::Mat& mat_p,cv::Point center)
{
    if(!mat_c.data)
    {
        dzlog_error("cart2polar img mat_c has no data ...");
        return false;
    }
    int rows_c = mat_c.rows;//这里通过传入的半径范围确定转换成的
    int cols_c = mat_c.cols;
    float polar_r = rows_c; // 原图半径
    float delta_r = polar_r / rows_c; //半径因子 //1
    float delta_t = 2.0*CV_PI / cols_c;  //角度因子 //1/r

    float center_polar_x = center.x;
    float center_polar_y = center.y;

    for (int i = 0; i < cols_c; i++)
    {
        float theta_p = i * delta_t;
        float sin_theta = std::sin(theta_p);
        float cos_theta = std::cos(theta_p);

        for (int j = 0; j < rows_c; j++)
        {
            float temp_r = j * delta_r; //放图第j行在圆图上对应的半径长度
            int polar_x = (int)(center_polar_x + temp_r * cos_theta);
            int polar_y = (int)(center_polar_y + temp_r * sin_theta);
            if(polar_x>=0||polar_y>=0||polar_x<mat_p.cols||polar_y<mat_p.rows)
            {
                mat_p.ptr<uchar>(polar_y)[polar_x] = mat_c.ptr<uchar>(j)[i];
            }
        }
    }
    return true;
}

bool DVAlgorithm::pointCart2polar(vector<Point> ptin, vector<Point>& ptout, cv::Point center, float radius)
{
    Point tmp(0, 0);
    //std::cout << "radius == " << radius << endl;
    for (int i = 0; i < ptin.size(); i++)
    {
        tmp.x = center.x + (float)ptin[i].y * cos((float)ptin[i].x /radius);
        tmp.y = center.y + (float)ptin[i].y * sin((float)ptin[i].x /radius);
        ptout.push_back(tmp);
    }
    return true;
}

bool DVAlgorithm::pointCart2polar(vector<Point> ptin,vector<Point>& ptout,cv::Point center,float radius,float thetaOffset)
{
    Point tmp(0, 0);
//    Point temp2(30, 20);
    //std::cout << "radius == " << radius << endl;
    for (int i = 0; i < ptin.size(); i++)
    {
        tmp.x = center.x + (float)ptin[i].y * cos(((float)ptin[i].x)/radius+thetaOffset);
        tmp.y = center.y + (float)ptin[i].y * sin(((float)ptin[i].x)/radius+thetaOffset);
        ptout.push_back(tmp);
    }
    return true;
}

//圆转方//还没有验证 已知笛卡尔坐标系下的坐标求极坐标的表示
bool DVAlgorithm::pointPolar2cart(vector<Point> ptin, vector<Point>& ptout, cv::Point center, float radius)
{
    float rho = 0.0;//角度
    float theta = 0.0;
    int rowY = 0;
    int colsX = 0;//弧长
    for (int i = 0; i < ptin.size(); i++)
    {
        rho = sqrt(pow(ptin[i].x - center.x,2) + pow(ptin[i].y - center.y,2));
        theta = atan2(ptin[i].y - center.y, ptin[i].x - center.x);
        rowY = ceil(rho);
        colsX = ceil(rho*theta);
        ptout.push_back(cv::Point(colsX, rowY));
    }
    return true;
}

//笛卡尔坐标系下的点转极坐标下的点  //方转圆
bool DVAlgorithm::pointCart2polar(cv::Point ptCart, cv::Point& ptPolar, cv::Point center, float radius)
{
    ptPolar.x = center.x + (double)ptCart.y * cos((double)ptCart.x / radius);
    ptPolar.y = center.y + (double)ptCart.y * sin((double)ptCart.x / radius);
    return true;
}
//笛卡尔坐标系下的点转极坐标下的点  //圆转方
bool DVAlgorithm::pointPolar2cart(cv::Point ptPolar, cv::Point& ptCart, cv::Point center, float radius)
{
    float rho = 0.0;//角度
    float theta = 0.0;
    int rowY = 0;
    int colsX = 0;//弧长
    rho = sqrt(pow(ptPolar.x - center.x, 2) + pow(ptPolar.y - center.y, 2));
    theta = atan2(ptPolar.y - center.y, ptPolar.x - center.x);
    rowY = ceil(rho);
    colsX = ceil(rho*theta);
    ptCart = cv::Point(colsX, rowY);
    return true;
}

//用reduce方法计算
void DVAlgorithm::getImgHmap_min(cv::Mat& imgsrc, map<int, int>& cartpos_Threshold)
{
    cv::Mat Hdist = Mat::zeros(imgsrc.rows, 1, CV_32FC1);//CV_32FC1
    reduce(imgsrc, Hdist, 1, CV_REDUCE_SUM, CV_32FC1);//水平方向投影
    Hdist = Hdist / imgsrc.cols;
    std::cout << "Hdist == " << Hdist << endl;
    float havg_min = 255.0;
    int row_min;
    for (int i = 0; i < Hdist.rows - 100; i++)
    {
        //std::cout << "Hdist.at<float>(i, 0) == "<<Hdist.at<float>(i, 0) << endl;
        if (Hdist.at<float>(i, 0) < havg_min)
        {
            row_min = i;
            havg_min = (float)Hdist.at<float>(i, 0);
        }
    }
    cartpos_Threshold.insert(pair<int, int>(row_min, havg_min));//水平最小平均灰度的行号和对应值
}

void DVAlgorithm::bwareaopen(cv::Mat& imgsrc,cCParam ccp)
{
    cv::Mat   labels, img_color, stats, centroids;
    ////1二值图像 2和原图一样大的标记图 //nccompsx5的矩阵 表示每个连通区域的外接矩形和面积(pixel)
    //centroids  nccompsx2的矩阵 表示每个连通区域的质心
    //nccomps  原图中连通区域数
    int nccomps = cv::connectedComponentsWithStats(
            imgsrc, labels,
            stats, centroids
    );
//    cout << "Total Connected Components Detected: " << nccomps << endl;
    dzlog_debug("Total Connected Components Detected: == %d",nccomps);
    for (int num = 1; num < nccomps; num++)
    {
        int STAT_AREA = stats.at<int>(num, CC_STAT_AREA);
        int STAT_WIDTH = stats.at<int>(num, CC_STAT_WIDTH);
        int STAT_HEIGHT = stats.at<int>(num, CC_STAT_HEIGHT);
        int STAT_LEFT = stats.at<int>(num, CC_STAT_LEFT);
        int STAT_TOP = stats.at<int>(num, CC_STAT_TOP);
        Rect temp(STAT_LEFT, STAT_TOP, STAT_WIDTH, STAT_HEIGHT);
        if (STAT_AREA < ccp.STAT_AREA || STAT_WIDTH<ccp.STAT_WIDTH || STAT_HEIGHT<ccp.STAT_HEIGHT)//TODO:这里关于最小连通域的阈值参数需要设置  外部输入
        {
            imgsrc(temp) = 0;
        }
    }
    //showImg("imgconnectedComponentsWithStats", imgsrc);
}

void DVAlgorithm::bwareaclose(cv::Mat& imgsrc,cCParam ccp)
{
    cv::Mat   labels, img_color, stats, centroids;
    ////1二值图像 2和原图一样大的标记图 //nccompsx5的矩阵 表示每个连通区域的外接矩形和面积(pixel)
    //centroids  nccompsx2的矩阵 表示每个连通区域的质心
    //nccomps  原图中连通区域数
    int nccomps = cv::connectedComponentsWithStats(
            imgsrc, labels,
            stats, centroids
    );
//    cout << "Total Connected Components Detected: " << nccomps << endl;
    dzlog_debug("Total Connected Components Detected: == %d",nccomps);
    for (int num = 1; num < nccomps; num++)
    {
        int STAT_AREA = stats.at<int>(num, CC_STAT_AREA);
        int STAT_WIDTH = stats.at<int>(num, CC_STAT_WIDTH);
        int STAT_HEIGHT = stats.at<int>(num, CC_STAT_HEIGHT);
        int STAT_LEFT = stats.at<int>(num, CC_STAT_LEFT);
        int STAT_TOP = stats.at<int>(num, CC_STAT_TOP);
        Rect temp(STAT_LEFT, STAT_TOP, STAT_WIDTH, STAT_HEIGHT);
        if (STAT_AREA > ccp.STAT_AREA || STAT_WIDTH > ccp.STAT_WIDTH || STAT_HEIGHT > ccp.STAT_HEIGHT)//TODO:这里关于最小连通域的阈值参数需要设置  外部输入
        {
            imgsrc(temp) = 0;
        }
    }
}

void DVAlgorithm::bwareaopen(Mat& imgsrc,int areaThreshold)
{
    dzlog_debug("get start bwareaopen ...");
    cv::Mat   labels, img_color, stats, centroids;
    ////1二值图像 2和原图一样大的标记图 //nccompsx5的矩阵 表示每个连通区域的外接矩形和面积(pixel)
    //centroids  nccompsx2的矩阵 表示每个连通区域的质心
    //nccomps  原图中连通区域数
    int nccomps = cv::connectedComponentsWithStats(
            imgsrc, labels,
            stats, centroids
    );
    for (int num = 1; num < nccomps; num++)
    {
        int STAT_AREA = stats.at<int>(num, CC_STAT_AREA);
        int STAT_WIDTH = stats.at<int>(num, CC_STAT_WIDTH);
        int STAT_HEIGHT = stats.at<int>(num, CC_STAT_HEIGHT);
        int STAT_LEFT = stats.at<int>(num, CC_STAT_LEFT);
        int STAT_TOP = stats.at<int>(num, CC_STAT_TOP);
        Rect temp(STAT_LEFT, STAT_TOP, STAT_WIDTH, STAT_HEIGHT);
        if (STAT_AREA < areaThreshold)//TODO:这里关于最小连通域的阈值参数需要设置  外部输入
        {
            imgsrc(temp) = 0;
        }
    }
}

void DVAlgorithm::getBounderInnerCoutours(cv::Mat& imgbin, vector<Point>& points, float innerRaidusEstimate, int OFFSET)
{
    int y_min = imgbin.rows;
    int y_max = 0;
    cv::Point ptTempUp = cv::Point(0, 0);
    cv::Point ptTempDown = cv::Point(0, 0);
    for (int i = 0; i < imgbin.cols; i++)
    {
        y_min = imgbin.rows;
        y_max = 0;
        for (int j = innerRaidusEstimate - OFFSET; j < innerRaidusEstimate + OFFSET; j++)
        {
            if ((int)imgbin.ptr<uchar>(j)[i] != 0)
            {
                if (y_min > j)
                {
                    y_min = j;
                }
                if (y_max < j)
                {
                    y_max = j;
                }
                //std::cout << "imgbin.ptr<uchar>(i)[j]" << (int)imgbin.ptr<uchar>(i)[j] << endl;
                //std::cout << "imgbin.at<uchar>(i,j)" << (int)imgbin.at<uchar>(i,j) << endl;
            }
        }
        if (y_min != imgbin.rows && y_max != 0)
        {
//            ptTempUp.x = i;
//            ptTempUp.y = y_min;
            ptTempDown.x = i;
            ptTempDown.y = y_max;
            //points.push_back(ptTempUp);
            points.push_back(cv::Point(i,y_max));
        }
    }
}

void DVAlgorithm::getImgHmap_min_filter(cv::Mat& src, map<int, int>& cartpos_Threshold)
{
    int height = src.rows;//高度
    int width = src.cols;//宽度
    MatrixXf imgsrc_matrix(height, width);
    //Eigen::Matrix<int, Dynamic, Dynamic> imgsrc_matrix;
    cv2eigen(src, imgsrc_matrix);
    Eigen::MatrixXf filter_x = MatrixXf::Ones(width, 1);
    Eigen::MatrixXf row_sum_matrix(height, 1);
    row_sum_matrix = imgsrc_matrix*filter_x;
    row_sum_matrix = row_sum_matrix / width;

    //std::cout << "横向投影   "<< row_sum_matrix << endl;
    float havg_min = 255.0;
    int row_min;
    for (int i = 0; i < (int)(height - 100); i++)
    {
        if (row_sum_matrix(i)< havg_min)
        {
            row_min = i;
            havg_min = row_sum_matrix(i);
        }
    }
    cartpos_Threshold.insert(pair<int, int>(row_min, havg_min));//水平最小平均灰度的行号和对应值
}

Mat DVAlgorithm::polyfit(vector<Point>& in_point, int n)
{
    int size = in_point.size();
    //所求未知数个数
    int x_num = n + 1;
    //构造矩阵U和Y
    Mat mat_u(size, x_num, CV_64F);
    Mat mat_y(size, 1, CV_64F);

    for (int i = 0; i < mat_u.rows; ++i)
        for (int j = 0; j < mat_u.cols; ++j)
        {
            mat_u.at<double>(i, j) = pow(in_point[i].x, j);
        }

    for (int i = 0; i < mat_y.rows; ++i)
    {
        mat_y.at<double>(i, 0) = in_point[i].y;
    }

    //矩阵运算 获得系数矩阵K
    Mat mat_k(x_num, 1, CV_64F);
    mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
//    cout << mat_k << endl;
    return mat_k;
}

void DVAlgorithm::getWhiteCircleContours(cv::Mat& imgbin, vector<cv::Point>& ptcontours)
{
    //showImg("imggetOutercontours", imgbin);
    vector<cv::Point> leftPts;
    vector<cv::Point> rightPts;
    int x_min=imgbin.cols;
    int x_max=0;
//    int y_min=imgbin.rows;
//    int y_max=0;
    cv::Point ptTempLeft = cv::Point(0, 0);
    cv::Point ptTempRight = cv::Point(0, 0);
    for (int i = 0; i < imgbin.rows; i++)
    {
        x_min = imgbin.cols;
        x_max = 0;
        for (int j = 0; j < imgbin.cols; j++)
        {
            if ((int)imgbin.ptr<uchar>(i)[j] != 0)
            {
                if (x_min > j)
                {
                    x_min = j;
                }
                if (x_max < j)
                {
                    x_max = j;
                }

                //std::cout << "imgbin.ptr<uchar>(i)[j]" << (int)imgbin.ptr<uchar>(i)[j] << endl;
                //std::cout << "imgbin.at<uchar>(i,j)" << (int)imgbin.at<uchar>(i,j) << endl;
            }
        }
        if (x_min != imgbin.cols && x_max != 0)
        {
            ptTempLeft.x = x_min;
            ptTempLeft.y = i;
            ptTempRight.x = x_max;
            ptTempRight.y = i;
            ptcontours.push_back(ptTempLeft);
            ptcontours.push_back(ptTempRight);
            //leftPts.push_back(ptTempLeft);
            //std::cout << "ptTempLeft == " << ptTempLeft << endl;
            //rightPts.push_back(ptTempRight);
            //std::cout << "ptTempRight == " << ptTempRight << endl;
        }
    }
    //std::cout << "good job ..." << endl;
}


void DVAlgorithm::getploy3Contours(vector<Point>& srcContours,vector<Point>& dstContours)
{
    Mat mat_k;
    mat_k = polyfit(srcContours, 3);
    //画出拟合曲线
    for (int i = srcContours[0].x; i < srcContours[srcContours.size() - 1].x; ++i) {
        Point2d ipt;
        ipt.x = i;
        ipt.y = 0;
        for (int j = 0; j < 3 + 1; ++j) {
            ipt.y += mat_k.at<double>(j, 0) * pow(i, j);
        }
        dstContours.push_back(ipt);
    }
}

void DVAlgorithm::getMinMaxYPoint(vector<Point>& pointsSrc,cv::Point& ptMax,cv::Point& ptMin)
{
    ptMax=pointsSrc[0];
    ptMin=pointsSrc[0];
    for(int i=1;i<pointsSrc.size();i++)
    {
        if(ptMax.y<pointsSrc[i].y)
        {
            ptMax=pointsSrc[i];
        }
        if(ptMin.y>pointsSrc[i].y)
        {
            ptMin=pointsSrc[i];
        }
    }
}

void DVAlgorithm::blurImg_gash(cv::Mat& imgsrc,Eigen::MatrixXf& filter,cv::Mat& imgout)
{
    showImg("imgbefore blur",imgsrc);
    cv::Mat kernel2;
    eigen2cv(filter, kernel2);
    //std::cout << filter1 << endl;
    cout<<"imgsrc.depth() == "<<imgsrc.depth()<<endl;
    cv::Mat imgconv;
    filter2D(imgsrc, imgout, -1, kernel2);
}

void DVAlgorithm::pointblur(vector<cv::Point> ptv, vector<cv::Point>& ptResult, int step)
{
    float y_step_avg=0.0;
    float y_step_sum = 0.0;
    Point ptavg;
    for (int i = 0; i < ceil(ptv.size()/step); i++)
    {
        y_step_sum = 0.0;
        for (int j = 0; j < step; j++)
        {
            y_step_sum += ptv[i*step + j].y;
        }
        y_step_avg = y_step_sum / (float)step;
        ptavg.x = i*step;
        ptavg.y = y_step_avg;
        ptResult.push_back(ptavg);
    }
}

void DVAlgorithm::bwareaopen(Mat& src,cCParam ccp,vector<cv::Rect>& results)
{
    Mat labels, stats, centroids;
    const int connectivity_8 = 8;
    int nLabels = connectedComponentsWithStats(src, labels, stats, centroids, connectivity_8, CV_32S);
    Rect result(0, 0, 0, 0);
    for (int num = 1; num < nLabels; num++)
    {
        int STAT_AREA = stats.at<int>(num, CC_STAT_AREA);
        int STAT_WIDTH = stats.at<int>(num, CC_STAT_WIDTH);
        int STAT_HEIGHT = stats.at<int>(num, CC_STAT_HEIGHT);
        int STAT_LEFT = stats.at<int>(num, CC_STAT_LEFT);
        int STAT_TOP = stats.at<int>(num, CC_STAT_TOP);
        Rect temp(STAT_LEFT, STAT_TOP, STAT_WIDTH, STAT_HEIGHT);
        if (STAT_AREA < ccp.STAT_AREA || STAT_HEIGHT < ccp.STAT_HEIGHT || STAT_WIDTH < ccp.STAT_WIDTH)
        {
            src(temp)=0;
            continue;
        }
        else
        {
            result = temp;
        }
        results.push_back(result);
    }
}

void DVAlgorithm::bwareaopen(Mat &src, Mat &dst, cCParam ccp,vector<cv::Rect>& results) {
    Mat dst2 = 0 * src.clone();
    Mat labels, stats, centroids;
    const int connectivity_8 = 8;
    int nLabels = connectedComponentsWithStats(src, labels, stats, centroids, connectivity_8, CV_32S);
    Rect result(0, 0, 0, 0);
    for (int num = 1; num < nLabels; num++)
    {
        int STAT_AREA = stats.at<int>(num, CC_STAT_AREA);
        int STAT_WIDTH = stats.at<int>(num, CC_STAT_WIDTH);
        int STAT_HEIGHT = stats.at<int>(num, CC_STAT_HEIGHT);
        int STAT_LEFT = stats.at<int>(num, CC_STAT_LEFT);
        int STAT_TOP = stats.at<int>(num, CC_STAT_TOP);
        Rect temp(STAT_LEFT, STAT_TOP, STAT_WIDTH, STAT_HEIGHT);
        if (STAT_AREA < ccp.STAT_AREA || STAT_HEIGHT < ccp.STAT_HEIGHT || STAT_WIDTH < ccp.STAT_WIDTH)
        {
            continue;
        }
        else
        {
            result = temp;
        }

        Mat target;
        compare(labels, num, target, CMP_EQ);
        dst2 = dst2 + target;
        results.push_back(result);
    }
    dst = dst2.clone();
}

//function：create a xml file
//param：xmlPath:xml文件路径
//return:0,成功，非0，失败
int DVAlgorithm::createXML(const char* xmlPath,const char* xmlRootName)
{
    const char* declaration ="<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>";
    XMLDocument doc;
    doc.Parse(declaration);//会覆盖xml所有内容

    //添加申明可以使用如下两行
    //XMLDeclaration* declaration=doc.NewDeclaration();
    //doc.InsertFirstChild(declaration);

    XMLElement* root=doc.NewElement(xmlRootName);
    doc.InsertEndChild(root);

    return doc.SaveFile(xmlPath);
}

void DVAlgorithm::getDistanceByHist(cv::Mat& imgtemplate,cv::Mat& imgtest,float& correlationDistance,float& bDistance,float& emdDistance)
{
    dzlog_info("开始异物检测 .... Hist 找四种差异数据,分别比较...");
    //直方图相似度比较
    vector<Mat> src;//迭代器push_back
    cv::Mat imgtempalte_bin;
    cv::Mat imgtest_bin;
//    cvtColor(imgtemplate,imgtempalte_bin,CV_RGB2GRAY);
//    cvtColor(imgtest,imgtest_bin,CV_RGB2GRAY);
//    cv::imshow("imgtemplate",imgtemplate);
//    cv::waitKey(0);
    src.push_back(imgtemplate);
//    temp = imread("roi_test_img.jpg", 1);
    src.push_back(imgtest);

    vector<Mat> hsv(2), hist(2),hist_img(2);
    int scale=10,histSize[] = { 8,8 }, ch[] = { 0,1 };//30rows,32cols
    float h_ranges[] = { 0,180 };
    float s_ranges[] = { 0,255 };
    const float* ranges[] = { h_ranges,s_ranges };
//    dzlog_info("src.size() == %d",src.size());
    for (int i = 0; i < 2 ; i++) {
//        dzlog_info("in for ...");
//        cv::imshow("src[i]",src[i]);
//        cv::imshow("hsv[i]",hsv[i]);
//        cv::waitKey(0);
        cvtColor(src[i], hsv[i], COLOR_RGB2HSV);
        calcHist(&hsv[i], 1, ch, noArray(), hist[i], 2, histSize, ranges, true);
        normalize(hist[i], hist[i], 0, 255, NORM_MINMAX);
        hist_img[i]=Mat::zeros(histSize[0] * scale, histSize[1] * scale, CV_8UC3);
        for (int h = 0; h < histSize[0]; h++) {
            for (int s = 0; s < histSize[1]; s++) {
                float hval = hist[i].at<float>(h, s);
                rectangle(hist_img[i], Rect(h * scale, s * scale, 10, 10), Scalar::all(hval), -1);
            }
        }
    }
    correlationDistance=compareHist(hist[0],hist[1],0);
    bDistance=compareHist(hist[0],hist[1],3);
    dzlog_info("直方图相关性比较 == %f",correlationDistance);
    dzlog_info("直方图巴氏距离 == %f",bDistance);
    //do EMD
    vector<Mat> sig(2);
    for (int i = 0; i < 2; i++) {
        vector<Vec3f> sigv;
        normalize(hist[i], hist[i], 1, 0, NORM_L1);
        for (int h = 0; h < histSize[0]; h++)
            for (int s = 0; s < histSize[1]; s++) {
                float hval = hist[i].at<float>(h, s);
                if (hval != 0)
                    sigv.push_back(Vec3f(hval, (float)h, (float)s));
            }
        sig[i] = Mat(sigv).clone().reshape(1);
        if (i > 0)
        {
            emdDistance=EMD(sig[0], sig[i], CV_DIST_L2);
            dzlog_info("图像相似性度量 EMD == %f",emdDistance);
        }
    }
}

void DVAlgorithm::DVsobel(cv::Mat imgsrc,cv::Mat& imgout)
{
    cv::Mat imgsrc2;
    imgsrc.convertTo(imgsrc2,CV_32FC1,1.0/255.0);
//    imgsrc.convertTo(imgsrc2,CV_16SC1,1.0/255.0);//TODO:这种情况没有试过
//    cout<<imgsrc2<<endl;
    Mat kernel_v = (Mat_<float>(3,3)<<-1, 0, 1, -2, 0, 2, -1, 0, 1);
//    cv::Mat grad_x,grad_y,grad_both;
//    Sobel(imgsrc, grad_x, imgsrc2.depth(), 0, 1, 3, 3, 1, BORDER_DEFAULT);
//    Sobel(imgsrc, grad_y, imgsrc2.depth(), 1, 0, 3, 3, 1, BORDER_DEFAULT);
//    Sobel(imgsrc, grad_both, imgsrc2.depth(), 1, 1, 3, 3, 1, BORDER_DEFAULT);
//    cout<<"grad_both"<<grad_both<<endl;
//    cv::imshow("grad_x",grad_x);
//    cv::imshow("grad_y",grad_y);
//    cv::imshow("grad_both",grad_both);
//    cv::waitKey(0);
//    Mat kernel_v = (Mat_<float>(3,3)<<-0.125, 0, 0.125, -0.25, 0, 0.25, -0.125, 0, 0.125);
    Mat kernel_h = (Mat_<float>(3,3)<<-1, -2, -1, 0, 0, 0, 1, 2, 1);
//    Mat kernel_h = (Mat_<float>(3,3)<<-0.125, -0.25, -0.125, 0, 0, 0, 0.125, 0.25, 0.125);
    cv::Mat dist_v=cv::Mat(imgsrc.rows,imgsrc.cols,CV_32FC1);
    cv::Mat dist_h=cv::Mat(imgsrc.rows,imgsrc.cols,CV_32FC1);
    filter2D(imgsrc2,dist_v,imgsrc2.depth(),kernel_v,Point(-1,-1),0.0);
//    cout<<dist_v<<endl;
    imshow("dist_v",dist_v);
    cv::waitKey(0);
    filter2D(imgsrc2,dist_h,imgsrc2.depth(),kernel_h);
    imshow("dist_h",dist_h);
    cv::waitKey(0);
    cv::Mat result = abs(dist_h)+abs(dist_v);
//    cout<<result<<endl;
    cv::imshow("result",result);
    cv::waitKey(0);

    cv::Mat b;
    b=dist_v.mul(dist_v)+dist_h.mul(dist_h);
    cv::Mat sumxy;
    sqrt( b, sumxy);
//    cout<<sumxy<<endl;
//    cout<<b<<endl;
//    cv::Mat e=cv::Mat::zeros(b.rows,b.cols,CV_8UC1);
//    cv::imshow("b",b);
//    cv::waitKey(0);
    //cv::Mat中均值
//    double minVal;
//    double maxVal;
//    Point minLoc;
//    Point maxLoc;
//
//    minMaxLoc( sumxy, &minVal, &maxVal, &minLoc, &maxLoc );
//
//    cout << "min val : " << minVal << endl;
//    cout << "max val: " << maxVal << endl;

//    double maxv;
//    minMaxLoc(sumxy,0,&maxv,0,0);
//    cout<<"maxvalue="<<maxv<<endl;
//    sumxy=sumxy/maxv*255.0;
//    cout<<sumxy<<endl;
//cout<<sumxy<<endl;
    double maxv;
    minMaxLoc(sumxy,0,&maxv,0,0);
//    cout<<"maxvalue="<<maxv<<endl;
    sumxy=sumxy/maxv * 255;
//    cout<<sumxy<<endl;
    imshow("sumxy",sumxy>100.0);
    cv::waitKey(0);
    cv::Mat imgoutTemp=sumxy>100.0;
    cv::imwrite("sumxy.jpg",imgoutTemp);
//    cv::Mat img_sobel;
//    Sobel(imgout, img_sobel, imgout.depth(), 1, 1, 3, 3, 1, BORDER_DEFAULT);
//    cv::imshow("imgsobel",img_sobel);
//    cv::waitKey(0);
    cv::Mat imgout2;
    imgoutTemp.convertTo(imgout2,CV_32FC1,1.0/255.0);

    cv::Mat dist2_v=cv::Mat(imgsrc.rows,imgsrc.cols,CV_32FC1);
    cv::Mat dist2_h=cv::Mat(imgsrc.rows,imgsrc.cols,CV_32FC1);

    Mat kernel_v2 = (Mat_<float>(2,2)<<1, 0, 0, -1);
    filter2D(imgout2,dist2_v,imgout2.depth(),kernel_v2);
//    cout<<dist_v<<endl;
    Mat kernel_h2 = (Mat_<float>(2,2)<<0, 1, -1, 0);
    filter2D(imgout2,dist2_h,imgout2.depth(),kernel_h2);
    cv::Mat result2 = abs(dist2_h)+abs(dist2_v);
    minMaxLoc(result2,0,&maxv,0,0);
    result2=result2/maxv * 255;
    imgout=result2 > 200;
//    cout<<result2<<endl;
    cv::imshow("result2",imgout);
    cv::waitKey(0);
//    cv::Mat dist;
//    cv::Canny(imgout,dist,100,255);
//    cv::imshow("dist",dist);
//    cv::waitKey(0);
//    sumxy.convertTo(imgout,CV_8UC1);
//    cout<<imgout<<endl;

//    cv:Scalar tempVal = cv::mean(b);
//    float matMean = tempVal.val[0]*4;
//    cout<<"matMean == "<< matMean<<endl;
//    int thresh=pow(matMean,0.5);// 4*2的平方根=2
//
//    cv::Mat result1;
//    cv::Mat result2;
//    result1=dist_h>thresh;
//    result2=dist_v>thresh;
//    cv::Mat dist;
//    addWeighted( result1, 1, result2, 1, 0.0,dist);
//    imgout=dist;
//    cv::imshow("result",dist);
//    cv::waitKey(0);

//    Mat mask_vertical = (cv::Mat_<uchar>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
//    Mat mask_horizontal = (cv::Mat_<uchar>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
//    Mat new_img = Mat::zeros(imgsobeltest.size(), CV_64F);
//    for (int i = 2; i < imgsobeltest.rows-2; i++)  // 规定当前像素为中间
//    {
//        for (int j = 2; j < imgsobeltest.cols-2; j++)
//        {
//            Mat block = imgsobeltest(Range(i - 1, i + 2), Range(j - 1, j + 2));
//            double vertical = block.dot(mask_vertical);
//            double horizontal = block.dot(mask_horizontal);
//            new_img.at<double>(i, j) = sqrt(pow(vertical, 2) + pow(vertical, 2));
//            //cout << new_img.at<double>(i, j) << endl;
//        }
//    }
//    normalize(new_img, new_img, 255, 0, NORM_MINMAX);
//    new_img.convertTo(new_img, CV_8UC1);
//    imshow("效果图", new_img);
//    cv::waitKey(0);

//    cv::Mat imgsobeltest=cv::imread("/home/hhg/Projects/DVProjects/imgs/6218/123.jpg");
//    cv::imshow("imgsrc",imgsobeltest);
//    cv::Mat imgsobel;
//    Sobel(imgsrc, imgsobel, imgsrc.depth(), 1, 1, 3, 3, 1, BORDER_DEFAULT);
////    cv::Mat imgbin;
////    cv::threshold(imgsobel,imgbin,150,255,CV_THRESH_BINARY);
//    cv::imshow("imgsobel",imgsobel>150);
////    cv::imshow("imgbin",imgbin);
//    cv::waitKey(0);
}

void DVAlgorithm::DVrobert(cv::Mat imgsrc,cv::Mat& imgout)
{
    cv::Mat dist2_v=cv::Mat(imgsrc.rows,imgsrc.cols,CV_32FC1);
    cv::Mat dist2_h=cv::Mat(imgsrc.rows,imgsrc.cols,CV_32FC1);
    double maxv;
    Mat kernel_v2 = (Mat_<float>(2,2)<<1, 0, 0, -1);
    filter2D(imgsrc,dist2_v,imgsrc.depth(),kernel_v2);
//    cout<<dist_v<<endl;
    Mat kernel_h2 = (Mat_<float>(2,2)<<0, 1, -1, 0);
    filter2D(imgsrc,dist2_h,imgsrc.depth(),kernel_h2);
    cv::Mat result2 = abs(dist2_h)+abs(dist2_v);
    minMaxLoc(result2,0,&maxv,0,0);
    result2=result2/maxv * 255;
    imgout=result2>200;
}


float DVAlgorithm::calcAvg(vector<Point>& oriPts)
{
    float sum;
    for(int i=0;i<oriPts.size();i++)
    {
        sum+=oriPts[i].y;
    }
    return sum/(float)oriPts.size();
}

//SSE(和方差)  该统计参数计算的是拟合数据和原始数据对应点的误差的平方和
float DVAlgorithm::calcSSE(vector<Point>& fitPts, vector<Point>& oriPts)
{
    float sse;
    for(int i=0;i<fitPts.size();i++)
    {
        sse+=pow(fitPts[i].y-oriPts[i].y,2);
    }
    return sse;
}
//Sum of squares of the regression 即预测数据与原始数据均值之差的平方和
float DVAlgorithm::calcSSR(vector<Point>& prePts,float avg_y)
{
    float ssr;
    for(int i=0;i<prePts.size();i++)
    {
        ssr+=pow(prePts[i].y-avg_y,2);
    }
    return ssr;
}
//Total sum of squares，即原始数据和均值之差的平方和
float DVAlgorithm::calcSST(vector<Point>& oriPts, float avg_y)
{
    float sst;
    for(int i=0;i<oriPts.size();i++)
    {
        sst+=pow(oriPts[i].y-avg_y,2);
    }
    return sst;
}



float DVAlgorithm::calcRsquare(float ssr,float sst)
{
    float Rsquare;
    Rsquare=ssr/sst;
    return Rsquare;
}

/**
src: source image (three channels)
bin: rectangle binarized image
plotDefectAreaMethod: global / local / shift
rOffset: radius offset
tOffset: theta offset. if tOffset > 0, it means that it shifts.
center: the center of the circle
rmax: the maximum radius
*/
void DVAlgorithm::plotDefectArea(cv::Mat& src, cv::Mat bin,
                                 plotDefectAreaMethod pdam, int rOffset,
                                 int tOffset, cv::Point Center, int rmax)
{
    cv::Point polarPoint;

    for (int i = 0; i < bin.cols; i++)
    {
        for (int j = 0; j < bin.rows; j++)
        {
            if (bin.ptr<uchar>(j)[i] > 200)
            {
                switch (pdam)
                {
                    case CV_GLOBAL_AREA:
                        if (tOffset)
                        {
                            if (i + tOffset < bin.cols)
                            {
                                pointCart2polar(cv::Point(i + tOffset, j), polarPoint, Center, rmax);
                            }
                            else
                            {
                                pointCart2polar(cv::Point(i - bin.cols + tOffset, j), polarPoint, Center, rmax);
                            }
                        }
                        else
                        {
                            pointCart2polar(cv::Point(i, j), polarPoint, Center, rmax);
                        }
                        break;

                    case CV_LOCAL_AREA:

                        if (tOffset)
                        {
                            if (i + tOffset < bin.cols)
                            {
                                pointCart2polar(cv::Point(i + tOffset, j + rOffset), polarPoint, Center, rmax);
                            }
                            else
                            {
                                pointCart2polar(cv::Point(i - bin.cols + tOffset, j + rOffset), polarPoint, Center, rmax);
                            }
                        }
                        else
                        {
                            pointCart2polar(cv::Point(i, j + rOffset), polarPoint, Center, rmax);
                        }
                        break;
                    default:
                        break;
                }
                src.at<cv::Vec3b>(polarPoint.y, polarPoint.x)[0] = 0;
                src.at<cv::Vec3b>(polarPoint.y, polarPoint.x)[1] = 255;
                src.at<cv::Vec3b>(polarPoint.y, polarPoint.x)[2] = 0;
            }
        }
    }
}

void DVAlgorithm::getRectsBorderPoints(vector<Rect> boundRects,vector<Point>& defPoints)
{
    for(int i=0;i<boundRects.size();i++)
    {
        for(int j=boundRects[i].x;j<boundRects[i].x+boundRects[i].width;j++)
        {
            defPoints.push_back(cv::Point(j,boundRects[i].y));//TODO:这里每个y后面都要加上一个原始对应的偏移值
            defPoints.push_back(cv::Point(j,boundRects[i].y+boundRects[i].height));
        }
        for(int k=boundRects[i].y;k<boundRects[i].y+boundRects[i].height;k++)
        {
            defPoints.push_back(cv::Point(boundRects[i].x,k));
            defPoints.push_back(cv::Point(boundRects[i].x+boundRects[i].width,k));
        }
    }
}

void DVAlgorithm::points2HoleCartImg(vector<Point> defPointsOnDetImg,int imgCartOffset,vector<Point>& defPointsOnHoleCart)
{
    for(int i=0;i<defPointsOnDetImg.size();i++)
    {
        defPointsOnHoleCart.push_back(cv::Point(defPointsOnDetImg[i].x,defPointsOnDetImg[i].y+imgCartOffset));
    }
}

void DVAlgorithm::points2HoleCartImg(vector<Point> defPointsOnDetImg,int imgCartYOffset,int imgCartXShiftOffset,
        int imgCols,vector<Point>& defPointsOnHoleCart)
{
    for(int i=0;i<defPointsOnDetImg.size();i++)
    {
        if(defPointsOnDetImg[i].x-imgCartXShiftOffset>=0)
        {
            defPointsOnHoleCart.push_back(cv::Point(defPointsOnDetImg[i].x-imgCartXShiftOffset,defPointsOnDetImg[i].y+imgCartYOffset));
        }else{
            defPointsOnHoleCart.push_back(cv::Point(imgCols+defPointsOnDetImg[i].x-imgCartXShiftOffset,defPointsOnDetImg[i].y+imgCartYOffset));
        }
    }
}

void DVAlgorithm::pointsHoleCart2ImgSrc(vector<Point> defPointsOnHoleCart,cv::Point center,int outerRadius,
        cv::Rect presetArea,vector<Point>& defPointsOnImgSrc)
{
    vector<Point> defPointsOnPreset;
    pointCart2polar(defPointsOnHoleCart,defPointsOnPreset,center,outerRadius);
    for(int i=0;i<defPointsOnPreset.size();i++)
    {
        defPointsOnImgSrc.push_back(cv::Point(defPointsOnPreset[i].x+presetArea.x,defPointsOnPreset[i].y+presetArea.y));
    }
}

void DVAlgorithm::drawDefPointsOnImgResult(cv::Mat& imgResult,vector<Point> defPoints)
{
    for(int i=0;i<defPoints.size();i++)
    {
        circle(imgResult,defPoints[i],2,Scalar(0,0,255));
    }
}