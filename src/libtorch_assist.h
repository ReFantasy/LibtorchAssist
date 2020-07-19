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
	 * @brief ����ת��Ϊ torch::Tensor
	 * @param array ����ָ��
	 * @param n �����С
	 * @return torch::Tensor
	*/
	torch::Tensor Array2Tensor(float* array, long long n);


	
	/**
	 * @brief OpenCV Mat ת��Ϊ torch::Tensor, Mat����Ϊ CV_8UC1 �� CV_8UC3����
	 * @param image cv::Mat
	 * @return torch::Tensor: (channels, rows, cols)
	*/
	torch::Tensor Mat2Tensor(const cv::Mat image);

	/**
	 * @brief torch::Tensor ת��Ϊ cv::Mat ,torch::Tensor's format:(channels, rows,cols)
	 * @param tensor_image torch::Tensor
	 * @return cv::Mat
	*/
	cv::Mat Tensor2Mat(const torch::Tensor &tensor_image);
}




#endif//_LIBTORCHASSIST_H__
