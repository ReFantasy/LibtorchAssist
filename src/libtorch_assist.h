//
// Created by TDL on 2020/7/11.
//

#ifndef _LIBTORCHASSIST_H__
#define _LIBTORCHASSIST_H__
#include <filesystem>
#include <vector>
#include <functional>

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


	/***************************************************************************************************
	 *
	 *    OpenCV Mat 和 Pytorch Tensor相互转换
	 *
	 ***************************************************************************************************/

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


	/***************************************************************************************************
	 * 
	 *    批量处理图像文件
	 *
	 ***************************************************************************************************/

	/**
	 * @brief 读取目录下的所有图片并以向量的形式返回，向量中的每个元素为一个tensor表示的图像
	 * @param path 读取目录
	 * @param flags 图像读取格式，0 读取灰度图，1读取彩色图
	 * @param img_op 图像操作，从磁盘读取图像后进行处理
	 * @return std::vector<torch::Tensor>
	*/
	std::vector<torch::Tensor> ReadImagesFromFolderToVector(std::filesystem::path path, int flags = 1, std::function<cv::Mat(cv::Mat)> img_op = nullptr);

	/**
	 * @brief 读取目录下的所有图片并返回单一的4维度tensor，[image_count, channels, rows, cols]
	 * @param path 读取目录
	 * @param flags 图像读取格式，0 读取灰度图，1读取彩色图
	 * @param img_op 图像操作，从磁盘读取图像后进行处理
	 * @return torch::Tensor
	*/
	torch::Tensor ReadImagesFromFolder(std::filesystem::path path, int flags = 1, std::function<cv::Mat(cv::Mat)> img_op = nullptr);

	/**
	 * @brief 读取目录下的所有图片并返回单一的4维度tensor，[image_count, channels, rows, cols]
	 * @param path 读取目录
	 * @param flags 图像读取格式，0 读取灰度图，1读取彩色图
	 * @param sz 将图像缩放到sz大小
	 * @return torch::Tensor
	*/
	torch::Tensor ReadImagesFromFolder(std::filesystem::path path, int flags = 1, cv::Size sz = {256,256});
	




}






#endif//_LIBTORCHASSIST_H__
