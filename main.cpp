#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#define _USE_MATH_DEFINES
#include <cmath>
#define M_PI 3.14159265358979323846
using namespace cv;
#define success_pad_count 5
#define key_measurement_leu @"measurement_leu" // 測定パッド(白血球)
#define key_measurement_pro @"measurement_pro" // 測定パッド(たんぱく質)
#define key_reference @"reference"             // リファレンスパッド
#define key_pad4 @"pad4"                       // PAD4(白PET)
#define key_upper_marker @"upper_marker"       // PAD5(黒PET)
#define key_lower_marker @"lower_marker"       // PAD位置決め、黒リファレンス
#define key_confirm @"confirm"                 // 試験紙全体（デバッグ用途）
class MyPixel{
    public:
     int red;
     int blue;
     int green;
     int sum;
    
};
void showImage(Mat image)
{
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);
    waitKey(0);
}
struct CGPoint
{
    float x;
    float y;
};

/* Sizes. */

struct CGSize
{
    float width;
    float height;
};
struct CGRect
{
    CGPoint origin;
    CGSize size;
};
// 位置補正の対象とするPad
class CorrectionDefinition
{
    double pitchAngle;
    double value;

public:
    CorrectionDefinition(double pitchAngleNew, double valueNew)
    {
        pitchAngle = pitchAngleNew;
        value = valueNew;
    }
};
enum CollectionPad
{
    Pad1,
    Pad2,
    Pad3,
    Pad4
};

double padPositionCorrectionValue(double pitchAngle, CollectionPad targetPad)
{
    std::vector<CorrectionDefinition> list;
    if (targetPad == Pad1)
        return 0.0;
    return 0.0;
};

Mat toGrayScaleImage(Mat image)
{
    cv::Mat *p = NULL;
    cv::Mat image_Mat;
    if (image.empty())
        return *p;
    cv::Mat gray_mat;
    cv::cvtColor(image, gray_mat, cv::COLOR_BGR2GRAY);
    if (gray_mat.empty())
        return *p;
    return gray_mat;
}
Mat toConstrastGrayImage(Mat image)
{
    // convert to gray image
    // AHE để cải thiện độ tương phản của ảnh
    // mức sáng và mức tối gần nhau nhất là step
    // max sáng và max tối có nhiều step thì hình ảnh càng sắc nét

    cv::Mat *p = NULL;
    if (image.empty())
        return *p;
    cv::Mat gray_mat;
    cv::cvtColor(image, gray_mat, COLOR_BGR2GRAY);
    if (gray_mat.empty())
        return *p;
    cv::Mat constrast_Mat;
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->apply(gray_mat, constrast_Mat);
    if (constrast_Mat.empty())
        return *p;
    return constrast_Mat;
}
bool checkBlurryImage(Mat image)
{
    // convert to GRAY
    // check ảnh có mờ hay không
    cv::Mat imageMat;
    // UIImageToMat(image, imageMat);
    if (imageMat.channels() == 1)
        return true;

    cv::Mat grayMat;
    cv::cvtColor(imageMat, grayMat, cv::COLOR_BGR2GRAY);

    cv::Mat laplacianMat;
    cv::Laplacian(grayMat, laplacianMat, CV_64F);
    cv::Scalar mean, stddev; // 0:1st channel, 1:2nd channel and 2:3rd channel
    meanStdDev(laplacianMat, mean, stddev);
    double variance = stddev[0] * stddev[0];
    // NSLog(@"variance = %f",variance);
    double threshold = 30;
    return (variance <= threshold);
}

Mat toFilter2DImage(Mat img_Mat)
{
    // áp dụng filter filter sharp
    // làm ảnh rõ hơn, sắc nét hơn, chói,
    // cv::Mat img_Mat;
    // UIImageToMat(image, img_Mat);

    // 先鋭化フィルタを作成する(4近傍か８近傍いまのところどっちか)
    const float k = -1.0;
    cv::Mat sharpningKernel4 = (cv::Mat_<float>(3, 3) << 0.0, k, 0.0, k, 5.0, k, 0.0, k, 0.0);
    // cv::Mat sharpningKernel8 = (cv::Mat_<float>(3, 3) << k, k, k, k, 9.0, k, k, k, k);

    // 先鋭化フィルタを適用する
    cv::Mat filter_Mat;
    cv::filter2D(img_Mat, filter_Mat, img_Mat.depth(), sharpningKernel4);
    return filter_Mat;
}
Mat toThresholdImage(Mat img_Mat, double thresh)
{
    // convert to gray
    // apply gaussian blur để giảm nhiễu giảm các chi tiết
    // app threshold để tìm các cạnh object trên nền màu đen
    // sau đó có thể áp dụng bitwise để đánh dấu duy nhất các pixel trong ảnh gốc mà mặt lạ có giá trị lớn hơn không
    // cv::Mat img_Mat;
    // UIImageToMat(image, img_Mat);

    cv::Mat gray_mat;
    cv::cvtColor(img_Mat, gray_mat, cv::COLOR_BGR2GRAY);

    cv::Mat blur_mat;
    cv::GaussianBlur(gray_mat, blur_mat, cv::Size(5, 5), 0, 0); // 0,0で自動計算
    // showImage(blur_mat);
    cv::threshold(blur_mat, blur_mat, thresh, 255, cv::THRESH_BINARY);
    return blur_mat;
}
Mat toBitwiseNotImage(Mat image_Mat)
{
    // đảo 0 với 1
    // đảo trắng vơi đen
    cv::Mat img_Mat;
    Mat *p = NULL;
    // UIImageToMat(image, img_Mat);
    if (img_Mat.empty())
        return *p;

    cv::bitwise_not(img_Mat, img_Mat); // 白黒の反転

    return image_Mat;
}
double calcAngle(cv::Point2f pointA, cv::Point2f pointB)
{
    // floor return largest integer not greater than x
    //  calculate angle between two vector
    // normalize it to the range [0, 2 π)
    //  convert to degree
    double r = atan2(pointB.y - pointA.y, pointB.x - pointA.x);
    if (r < 0)
    {
        r = r + 2 * M_PI;
    }
    return floor(r * 360 / (2 * M_PI));

    return 0.0;
}
bool judgeBlackMarker(std::vector<cv::Point> point, Mat raw_Mat, bool isCompare)
{
    // trích tứ giác
    // làm thẳng đường viền
    //
    double a = contourArea(point, false); // diện tích contour
    if (a > 2500)
    {
        // 小さすぎる
        return false;
    }

    // //輪郭を直線近似してみて四角のものだけを抽出
    // // trích chỉ cái tứ giác, làm đường bao xấp xỉ đường thẳng

    // std::vector<cv::Point> approx;
    // cv::approxPolyDP(cv::Mat(point), approx, 0.01 * cv::arcLength(point, true), true);
    // if (approx.size() < 4 || approx.size() > 8)
    // {
    //     // 形状が合わない
    //     return false;
    // }

    // // NSLog(@"a = %f",a);

    // cv::RotatedRect rotateRect = cv::minAreaRect(point);

    // double longSide = rotateRect.size.width > rotateRect.size.height ? rotateRect.size.width : rotateRect.size.height;
    // double shortSide = rotateRect.size.width > rotateRect.size.height ? rotateRect.size.height : rotateRect.size.width;
    // // NSLog(@"longSide = %f shortSide = %f",longSide,shortSide);

    // if (isCompare)
    // {
    //     // 矩形の幅はフレームサイズ幅*0.55以上　ShortSideはlongSide*0.50以上
    //     if (longSide < (double)(raw_Mat.cols * 0.55) || (shortSide < (double)(longSide * 0.50)))
    //     {
    //         // カメラ離れすぎ　または　細長すぎのため除外
    //         return false;
    //     }
    // }

    // if (rotateRect.size.width < rotateRect.size.height)
    // {
    //     cv::swap(rotateRect.size.width, rotateRect.size.height);
    //     rotateRect.angle += 90.f;
    // }

    // if (isCompare)
    // {
    //     if (rotateRect.size.width >= raw_Mat.cols - 1 || rotateRect.size.height >= raw_Mat.rows - 1)
    //     {
    //         // 撮影枠にピッタリすぎる（カメラ近すぎ）ため除外
    //         return false;
    //     }
    // }
    return true;
}
Mat drawRectOnImage(Mat image, Rect rect)
{
    RNG rng(12345);
    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    rectangle(image, rect, color);
    return image;
}
void drawRotateRect(RotatedRect rotateRect, Mat drawing)
{
    RNG rng(12345);
    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    Point2f rect_points[4];
    rotateRect.points(rect_points);
    for (int j = 0; j < 4; j++)
    {
        line(drawing, rect_points[j], rect_points[(j + 1) % 4], color);
    }
}

bool compareContour(std::vector<cv::Point> contourA, std::vector<cv::Point> contourB)
{
    cv::RotatedRect rectA = cv::minAreaRect(contourA);
    cv::RotatedRect rectB = cv::minAreaRect(contourB);
    return rectA.center.x < rectB.center.x;
}
bool functionRemoveContour(std::vector<cv::Point> contour)
{
    double area = contourArea(contour);
    if (area < 2500)
        return true;
    return false;
}
std::vector<std::vector<cv::Point>> convertContourToRect(std::vector<std::vector<cv::Point>> contours)
{

    sort(contours.begin(), contours.end(), compareContour);

    return contours;
}

#define DEGREES_TO_RADIANS(__ANGLE__) ((__ANGLE__) / 180.0 * M_PI)
int thresh = 100;
RNG rng(12345);

Mat getSensorPartImage(Mat threshold_Mat, Mat raw_Mat, int position)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    Mat p = Mat::zeros(threshold_Mat.size(), CV_8UC3);

    // cv::Mat threshold_Mat;
    // UIImageToMat(thresholdImage, threshold_Mat);

    // cv::Mat raw_Mat; //これはフレームサイズとしても利用できる
    // UIImageToMat(rawImage, raw_Mat);

    // NSLog(@"threshold_Mat.cols = %d threshold_Mat.rows = %d",threshold_Mat.cols,threshold_Mat.rows);
    cv::findContours(threshold_Mat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_L1);
    Mat drawing = Mat::zeros(threshold_Mat.size(), CV_8UC3);

    contours = convertContourToRect(contours);
    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    std::vector<std::vector<cv::Point>> listMinRotatedRect;
    int numberElement = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        if (functionRemoveContour(contours[i]))
        {
            listMinRotatedRect.push_back(contours[i]);
        }
        drawContours(drawing, contours, (int)i, color);
    }
    // 黒マーカー（2個）を探す
    cv::RotatedRect marker1;
    cv::RotatedRect marker2;

    for (int i = position; i < position + 2; i++)
    {
        cv::RotatedRect rotateRect = cv::minAreaRect(listMinRotatedRect[i]);
       
        // contour

        bool judgeResult = judgeBlackMarker(listMinRotatedRect[i], raw_Mat, false);
        if (judgeResult == false)
        {
            continue;
        }
        
        drawRotateRect(rotateRect, drawing);

        if (marker1.size.width == 0)
        {
            marker1 = rotateRect;
        }
        else if (marker2.size.width == 0)
        {
            marker2 = rotateRect;
        }
        else
        {
            if (rotateRect.center.y > marker1.center.y ||
                rotateRect.center.y > marker2.center.y)
            {
                if (marker1.center.y < marker2.center.y)
                {
                    marker1 = rotateRect;
                }
                else
                {
                    marker2 = rotateRect;
                }
            }
        }
    }
    
   
    cv::RotatedRect upperMarkerRect;
    cv::RotatedRect lowerMarkerRect;
    if (marker1.center.y < marker2.center.y)
    {
        upperMarkerRect = marker1;
        lowerMarkerRect = marker2;
    }
    else
    {
        upperMarkerRect = marker2;
        lowerMarkerRect = marker1;
    }

    cv::Rect padUpRect(upperMarkerRect.center.x-upperMarkerRect.size.width/2, upperMarkerRect.center.y-upperMarkerRect.size.height/2, upperMarkerRect.size.width, upperMarkerRect.size.height);
    cv::Rect padLowRect(lowerMarkerRect.center.x - lowerMarkerRect.size.width/2,lowerMarkerRect.center.y-lowerMarkerRect.size.height/2,lowerMarkerRect.size.width,lowerMarkerRect.size.height);
    double widthAllRect=upperMarkerRect.size.width;
    double heightAllRect = lowerMarkerRect.center.y-upperMarkerRect.center.y+lowerMarkerRect.size.height/2+upperMarkerRect.size.height/2;
    double originAllRectX=upperMarkerRect.center.x-upperMarkerRect.size.width/2;
    double originAllRectY=upperMarkerRect.center.y-upperMarkerRect.size.height/2;
    double padding =5;
    cv::Rect allRect(originAllRectX-padding,originAllRectY-padding,widthAllRect+2*padding,heightAllRect+2*padding);
    cv::Mat mat=raw_Mat(allRect);
    int m=0 ; // number pad on between
    if(position==0){
        m= 14;
    }
    else if (position ==2){
        m=13;
    }
    else if( position ==4){
        m=12;
    }
    float h = upperMarkerRect.size.height;
    float ya = originAllRectY;
    float yb = lowerMarkerRect.center.y-lowerMarkerRect.size.height/2;
    float distanceBetweenPad=(yb - ya - (m-2)*h)/ (m-1);

 
    double xn=originAllRectX;
    for(int i=0;i<m;i++)
    {
        double yn=ya+i*(distanceBetweenPad+h);
        double xn=originAllRectX;
        drawRectOnImage(raw_Mat, Rect(xn,yn,upperMarkerRect.size.width,upperMarkerRect.size.height));
        
    }

    return mat;
}

CGRect *getUpperBlackMarkerPosition(Mat threshold_Mat, Mat raw_Mat)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // cv::Mat threshold_Mat;
    // UIImageToMat(thresholdImage, threshold_Mat);

    // cv::Mat raw_Mat; //これはフレームサイズとしても利用できる
    // UIImageToMat(sensorPartImage, raw_Mat);

    // NSLog(@"threshold_Mat.cols = %d threshold_Mat.rows = %d",threshold_Mat.cols,threshold_Mat.rows);
    cv::findContours(threshold_Mat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_L1);

    // int imageWidth =     sensorPartImage.size.width;
    // int imageHeight =     sensorPartImage.size.height;
   Mat drawing = Mat::zeros(threshold_Mat.size(), CV_8UC3);
   Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
   

    cv::RotatedRect pad4Rect;  //検出したPad4黒マーカー
    cv::RotatedRect underRect; //検出した下方黒マーカー
    for (int i = 0; i < contours.size(); i++)
    {
        drawContours(drawing, contours, (int)i, color);
        bool judgeResult = judgeBlackMarker(contours[i], raw_Mat, false);
        if (judgeResult == false)
        {
            continue;
        }

        cv::RotatedRect rotateRect = cv::minAreaRect(contours[i]);

        double aspect = rotateRect.size.width / rotateRect.size.height;
        if (aspect < 0.9 || aspect > 1.5)
        {
            // 斜めから撮影されることを考慮しても形がおかしいため除外
            continue;
        }

        if (rotateRect.center.y < (raw_Mat.rows / 2))
        {
            // 真ん中よりも上にある
            if (pad4Rect.size.width == 0)
            {
                pad4Rect = rotateRect;
            }
            else if (rotateRect.center.y > pad4Rect.center.y)
            {
                // 下のほうにあるものを優先する（Pad1を黒マーカーとして拾ってしまった場合にPad4を優先するため）
                pad4Rect = rotateRect;
            }
        }
        else
        {
            // 真ん中よりも下にある
            if (underRect.size.width == 0)
            {
                underRect = rotateRect;
            }
            else if (rotateRect.center.y < underRect.center.y)
            {
                // 下のほうにあるものを優先する（下方黒マーカーは一番下にあるので）
                underRect = rotateRect;
            }
        }
    }
    showImage(drawing);
    CGRect *rectArray = new CGRect[2];
    rectArray[0].origin.x = pad4Rect.center.x - (pad4Rect.size.width / 2);
    rectArray[0].origin.y = pad4Rect.center.y - (pad4Rect.size.height / 2);
    rectArray[0].size.width = pad4Rect.size.width;
    rectArray[0].size.height = pad4Rect.size.height;

    rectArray[1].origin.x = underRect.center.x - (underRect.size.width / 2);
    rectArray[1].origin.y = underRect.center.y - (underRect.size.height / 2);
    rectArray[1].size.width = underRect.size.width;
    rectArray[1].size.height = underRect.size.height;
    return rectArray;
}

std::vector<Mat> getPadImages(Mat thresholdImage, double threshold, Mat raw_Image, double pitchAngle)
{
    std::vector<Mat> pads = {};
    Mat sensorPartImageTemp = getSensorPartImage(thresholdImage, raw_Image, 4);
    Mat sensorPartImage = sensorPartImageTemp;
    showImage(sensorPartImage);
    // Mat sensorPartImageTemp2 = getSensorPartImage(thresholdImage, raw_Image, 2);
    // showImage(sensorPartImageTemp2);
    // Mat sensorPartImageTemp3 = getSensorPartImage(thresholdImage, raw_Image, 4);
    // showImage(sensorPartImageTemp3);
    Mat sensorPartImageThreshold = toThresholdImage(sensorPartImage, 130);
    showImage(sensorPartImageThreshold);
    CGRect *blackRects = getUpperBlackMarkerPosition(sensorPartImageThreshold, sensorPartImage);

    // int ya; //upper pad
    // int yb; // lower pad 
   
    return pads;
}
bool comparePixel(MyPixel p1, MyPixel p2){
    return p1.sum < p2.sum;
}
MyPixel getPadColor(Mat image)
{
    double startX = image.rows * 0.35;
    double endX = image.rows * 0.65;
    double startY = image.cols * 0.35;
    double endY = image.cols * 0.65;
    double numberPoint = 3;
    double intervalX = (endX - startX) * (numberPoint - 1);
    double intervalY = (endY - startY) * (numberPoint - 1);
    int y = 0;
    int countOfY = 0;
    int numberPixel=0;
   
    std::vector<MyPixel> listPixels;
    for (countOfY = 0; countOfY < numberPoint; countOfY++)
    {
        for (int countOfX = 0; countOfX < numberPoint; countOfX++)
        {
            Vec3b pixel = image.at<Vec3b>(startX + intervalX * countOfX, startY + intervalY * countOfY);
            int r =pixel(2);
            int g = pixel(1);
            int b = pixel(0);
            MyPixel myPixel;
            myPixel.red=r;
            myPixel.green=g;
            myPixel.blue=b;
            myPixel.sum=r+g+b;
            listPixels.push_back(myPixel);
        }
    }
    sort(listPixels.begin(),listPixels.end(),comparePixel);
    int exclusion=2;
    int sumRed=0;
    int sumBlue=0;
    int sumGreen=0;
    for (int i = exclusion; i < listPixels.size()-exclusion; i++)
    {
        sumRed+=listPixels[i].red;
        sumBlue+=listPixels[i].blue;
        sumGreen+=listPixels[i].green;
    }
    int averageRed=sumRed/(listPixels.size()-exclusion*2);
    int averageBlue=sumBlue/(listPixels.size()-exclusion*2);
    int averageGreen=sumGreen/(listPixels.size()-exclusion*2);
    MyPixel averagePixel;
    averagePixel.red=averageRed;
    averagePixel.blue=averageBlue;
    averagePixel.green=averageGreen;
    averagePixel.sum=averageRed+averageBlue+averageGreen;
    return averagePixel;
    

}
int main(int argc, char **argv)
{

    Mat image;
    image = imread("D:/projectopencvtool/lenna15.png");
    int width = image.rows;
    int height = image.cols;
    // BGR
    int numbercount = 3;
    int interval = numbercount - 1;
    // std::cout << width<< height<<" "<<image.at<Vec3b>(400,100)<<std::endl;
    int intervalX = width / interval;
    int intervalY = width / interval;
    // for (int r = 0; r < image.rows; ++r)
    // {
    //     for (int c = 0; c < image.cols; ++c)
    //     {
    //         std::cout << "Pixel at position (x, y) : (" << c << ", " << r << ") =" << image.at<Vec3b>(r, c) << std::endl;
    //     }
    // }
    // imwrite("threshold.jpg", toThresholdImage(image,80.0));

    // image= toFilter2DImage(image);
    int threshold=3;
    cv::Mat thresholdImage= toThresholdImage(image,threshold);
    showImage(thresholdImage);
    double pitchAngle = 0.0;
    std::vector<cv::Mat> info = getPadImages(thresholdImage, threshold, image, pitchAngle);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    return 0;
}

