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
	 * @brief ����ת��Ϊ torch::Tensor
	 * @param array ����ָ��
	 * @param n �����С
	 * @return torch::Tensor
	*/
	torch::Tensor Array2Tensor(float* array, long long n);


	/***************************************************************************************************
	 *
	 *    OpenCV Mat �� Pytorch Tensor�໥ת��
	 *
	 ***************************************************************************************************/

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


	/***************************************************************************************************
	 * 
	 *    ��������ͼ���ļ�
	 *
	 ***************************************************************************************************/

	/**
	 * @brief ��ȡĿ¼�µ�����ͼƬ������������ʽ���أ������е�ÿ��Ԫ��Ϊһ��tensor��ʾ��ͼ��
	 * @param path ��ȡĿ¼
	 * @param flags ͼ���ȡ��ʽ��0 ��ȡ�Ҷ�ͼ��1��ȡ��ɫͼ
	 * @param img_op ͼ��������Ӵ��̶�ȡͼ�����д���
	 * @return std::vector<torch::Tensor>
	*/
	std::vector<torch::Tensor> ReadImagesFromFolderToVector(std::filesystem::path path, int flags = 1, std::function<cv::Mat(cv::Mat)> img_op = nullptr);

	/**
	 * @brief ��ȡĿ¼�µ�����ͼƬ�����ص�һ��4ά��tensor��[image_count, channels, rows, cols]
	 * @param path ��ȡĿ¼
	 * @param flags ͼ���ȡ��ʽ��0 ��ȡ�Ҷ�ͼ��1��ȡ��ɫͼ
	 * @param img_op ͼ��������Ӵ��̶�ȡͼ�����д���
	 * @return torch::Tensor
	*/
	torch::Tensor ReadImagesFromFolder(std::filesystem::path path, int flags = 1, std::function<cv::Mat(cv::Mat)> img_op = nullptr);

	/**
	 * @brief ��ȡĿ¼�µ�����ͼƬ�����ص�һ��4ά��tensor��[image_count, channels, rows, cols]
	 * @param path ��ȡĿ¼
	 * @param flags ͼ���ȡ��ʽ��0 ��ȡ�Ҷ�ͼ��1��ȡ��ɫͼ
	 * @param sz ��ͼ�����ŵ�sz��С
	 * @return torch::Tensor
	*/
	torch::Tensor ReadImagesFromFolder(std::filesystem::path path, int flags = 1, cv::Size sz = {256,256});
	




}






#endif//_LIBTORCHASSIST_H__
