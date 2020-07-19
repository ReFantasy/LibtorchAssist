//
// Created by TDL on 2020/7/11.
//

#ifndef _LIBTORCHASSIST_H__
#define _LIBTORCHASSIST_H__

namespace torch {
	class Tensor;
}
namespace cv {
	class Mat;
}

namespace LibtorchAsssst
{
	/**
	 * @brief 数组转化为 torch::Tensor
	 * @param array 数组指针
	 * @param n 数组大小
	 * @return torch::Tensor
	*/
	torch::Tensor Array2Tensor(float* array, long long n);


	
	/**
	 * @brief OpenCV Mat 转化为 torch::Tensor, Mat必须为 CV_8UC1 或 CV_8UC3类型
	 * @param image cv::Mat
	 * @return torch::Tensor: (channels, rows, cols)
	*/
	torch::Tensor Mat2Tensor(const cv::Mat image);

	/**
	 * @brief torch::Tensor 转化为 cv::Mat ,torch::Tensor's format:(channels, rows,cols)
	 * @param tensor_image torch::Tensor
	 * @return cv::Mat
	*/
	cv::Mat Tensor2Mat(const torch::Tensor &tensor_image);
}




#endif//_LIBTORCHASSIST_H__
