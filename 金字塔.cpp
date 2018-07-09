#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

/// 全局变量
Mat src, dst, tmp,sub,tem,dst1;
char* window_name = "Pyramids Demo";


/**
 * @函数 main
 */
//该函数用于构建高斯金字塔
int nOctaveLayers=3;
double sigma=1.6;
void buildGaussianPyramid(Mat& base, vector<Mat>& pyr, int nOctaves )
{
	//向量数组 sig 表示每组中计算各层图像所需的方差，加三是因为高斯金字塔层数等于S+3（DOG的是S+2）
    vector<double> sig(nOctaveLayers + 3);
	//定义高斯金字塔的总层数，nOctaves*(nOctaveLayers + 3)即组数×层数
    pyr.resize(nOctaves*(nOctaveLayers + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
	//提前计算好各层图像所需的方差
    sig[0] = sigma;//第一层图像的尺度为基准层尺度 σ0 
	//k等于2的s分之一次方
    double k = pow( 2., 1. / nOctaveLayers );
	//遍历所有层，计算方差
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
		//由公式 10 计算前一层图像的尺度
        double sig_prev = pow(k, (double)(i-1))*sigma;
		//由公式 10 计算当前层图像的尺度
        double sig_total = sig_prev*k;
		//计算公式 4 中高斯函数所需的方差，并存入 sig 数组内
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }
	//遍历高斯金字塔的所有层，构建高斯金字塔（这里同样大小的称为一组，每一组中包含不同层（模糊程度不同）
    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 3; i++ )
        {
			//dst 为当前层图像矩阵
            Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
			//如果当前层为高斯金字塔的第 0 组第 0 层，则直接赋值
            if( o == 0  &&  i == 0 )
                dst = base;//把由 createInitialImage 函数得到的基层图像矩阵赋予该层
            // base of new octave is halved image from end of previous octave
			//如果当前层是除了第 0 组以外的其他组中的第 0 层，则要进行降采样处理
            else if( i == 0 )
            {
				//提取出当前层所在组的前一组中的倒数第 3 层图像（因为高斯金字塔中每一组的第一张图片都和上一组倒数第三张图片的清晰度相同）
                const Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
				//隔点降采样处理
                resize(src, dst, Size(src.cols/2, src.rows/2),
                       0, 0, INTER_NEAREST);
            }
			//除了以上两种情况以外的其他情况的处理
            else
            {
				//提取出当前层的前一层图像
                const Mat& src = pyr[o*(nOctaveLayers + 3) + i-1];
				//由前一层尺度图像得到当前层的尺度图像（用公式L(x,y,sigma)=G(x,y,sigema)*I(x,y))
                GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
			imshow("pyr",dst);
			waitKey(1000);
        }
    }
}
//该函数用于构建 DoG 金字塔
void buildDoGPyramid( vector<Mat>& gpyr, vector<Mat>& dogpyr )
{
	//计算金字塔的组的数量
    int nOctaves = (int)gpyr.size()/(nOctaveLayers + 3);
	//定义 DoG 金字塔的总层数，DoG 金字塔比高斯金字塔每组少一层
    dogpyr.resize( nOctaves*(nOctaveLayers + 2) );
	//遍历 DoG 的所有层，构建 DoG 金字塔
    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 2; i++ )
        {
			//提取出高斯金字塔的当前层图像
            const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
			//提取出高斯金字塔的下层图像
            const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
			//提取出 DoG 金字塔的当前层图像
            Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
			//DoG 金字塔的当前层图像等于高斯金字塔的当前层图像减去高斯金字塔的上层图像
            subtract(src2, src1, dst, noArray(), CV_32F);
			//cout << dst << endl;
			imshow("dog",dst);
			waitKey(1000);
        }
    }
}

 Mat createInitialImage( Mat& img, bool doubleImageSize, float sigma )
{
    Mat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )//如果是多通道图像
        cvtColor(img, gray, COLOR_BGR2GRAY);        //先转化为灰度图
    else
        img.copyTo(gray);

	//调整图像的像素数据类型
    gray.convertTo(gray_fpt, -1, 1, 0);
    float sig_diff;
	float SIFT_INIT_SIGMA=0.5;
    if( doubleImageSize )//如果给出了doubleImageSize参数，则需要扩大图像的长宽尺寸
    {
		//sig_diff 为的高斯函数左边分数的分母上所需要的方差
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );//后面的f是以float类型处理的意思
																								   //SIFT_INIT_SIGMA默认值是0.5
																								   //sigma是自己设定的值
        Mat dbl;
		//利用双线性插值法把图像的长宽都扩大 2 倍
        resize(gray_fpt, dbl, Size(gray.cols*2, gray.rows*2), 0, 0, INTER_LINEAR);
		//对图像进行高斯平滑处理
        GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
        return dbl;//输出图像矩阵
    }
    else//如果未给出doubleImageSize参数，则不需要扩大图像的尺寸
    {
		//sig_diff 为的高斯函数左边分数的分母上所需要的方差
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
		//对图像进行高斯平滑处理
        GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
        return gray_fpt;//输出图像矩阵
    }
}

int main( int argc, char** argv )
{
  src = imread( "图片的路径");
  if( !src.data )
    { printf(" No data! -- Exiting the program \n");
      return -1; }
  Mat base=createInitialImage(src,0,1.6);
  imshow("第一张",base);
  waitKey();
  vector<Mat> pyr;//高斯金字塔图片
  vector<Mat> dog;//差分金字塔图像
  buildGaussianPyramid(base,pyr,3);
  buildDoGPyramid(pyr,dog);
  //src表示原始图像，tmp表示每组第0层，dst表示模糊后的图像，tem表示每组倒数第三张图片
  //tmp = src;
  //dst = tmp;
  //tem = dst;

  /// 创建显示窗口
 /* namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  imshow( window_name, dst );
  waitKey(1000);

  /// 循环
  for(int i = 0;i<4;i++)
  {
	  for(int j =0;j<5;j++)
	  {
		  GaussianBlur(tmp,dst,Size(3,3),j,j);
		  string window_name = format("第%d组第%d层",i,j);
		  imshow( window_name, dst );
		  waitKey(1000);
		  if(j==2)
			  tem=dst;
	  }
	  pyrDown( tem, dst);
	  tmp = dst;
  }
  return 0;*/
}
