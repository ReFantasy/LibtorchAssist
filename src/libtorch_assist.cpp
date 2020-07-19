//
// Created by TDL on 2020/7/11.
//
#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "libtorch_assist.h"

namespace LibtorchAsssst
{
	

	torch::Tensor Array2Tensor(float* array, long long n)
	{
		auto tensor = torch::from_blob(array, { n }, torch::kFloat);
		return tensor.clone();
	}

	torch::Tensor Mat2Tensor(const cv::Mat image)
	{
		if ((image.type() == CV_8UC1) || (image.type() == CV_8UC3))
		{
			int channels = image.channels();
			auto tensor_image = torch::from_blob(image.data, { channels, image.rows, image.cols }, at::kByte); 
			return tensor_image.clone();
		}
		return torch::empty({0});
	}

	cv::Mat Tensor2Mat(const torch::Tensor& tensor_image)
	{
		// tensor_image should be the format of (channels, rows,cols)
		assert(tensor_image.dim()==3);

		int channels = tensor_image.size(0);
		int rows = tensor_image.size(1);
		int cols = tensor_image.size(2);

		cv::Mat img;
		if (channels == 3)
			img = cv::Mat(rows, cols, CV_8UC3);
		else
			img = cv::Mat(rows, cols, CV_8UC1);

		auto tmp_tensor = tensor_image.to(torch::kByte);
		memcpy(img.data, tmp_tensor.data<uchar>(), channels * rows * cols);

		return img;
	}

	std::vector<torch::Tensor> ReadImagesFromFolderToVector(std::filesystem::path path, int flags /*= 1*/, std::function<cv::Mat(cv::Mat)> img_op/* nullptr */)
	{
		// 判断路径是否存在
		if (!std::filesystem::exists(path))
		{
			throw "path is not exist!";
		}

		std::vector<torch::Tensor> tensors;
		auto file_lists = std::filesystem::directory_iterator(path);
		for (auto& file : file_lists)
		{
			// 读取当前目录下的所有非目录文件
			if (file.status().type() != std::filesystem::file_type::directory)
			{
				auto cv_image = cv::imread(file.path().generic_string(), flags);

				if (img_op)
				{
					cv_image = img_op(cv_image);
				}

				auto tensor_image = Mat2Tensor(cv_image);
				tensors.push_back(tensor_image);
			}
		}

		return tensors;
	}

	torch::Tensor ReadImagesFromFolder(std::filesystem::path path, int flags, std::function<cv::Mat(cv::Mat)> img_op)
	{
		
		auto tensors = ReadImagesFromFolderToVector(path, flags, img_op);

		// 增加一个维度并拼接
		auto _tensors = torch::stack(tensors);
		return _tensors;
	}



	torch::Tensor ReadImagesFromFolder(std::filesystem::path path, int flags /*= 1*/, cv::Size sz)
	{
		auto img_op = [sz](cv::Mat img)
		{
			cv::resize(img, img, sz);
			return img;
		};
		return ReadImagesFromFolder(path, flags, img_op);
	}

}

