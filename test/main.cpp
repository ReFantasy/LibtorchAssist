#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "libtorch_assist.h"
#include <iostream>
#include <filesystem>

using namespace std;
int main()
{
	std::filesystem::path images_path("F:\\mixtrain\\input");
	
	auto img_op = [](cv::Mat img)
	{
		cv::resize(img, img, { 200,200 });
		return img;
	};

	auto images_tensor = LibtorchAsssst::ReadImagesFromFolder(images_path, 1, img_op);

	std::cout << images_tensor.sizes() << std::endl;

	auto m1 = LibtorchAsssst::Tensor2Mat(images_tensor.index({ 1 }));

	cv::imshow("1", m1);

	cv::waitKey();


	/*auto image1 = cv::imread("F:/mixtrain/label/1.jpg", 1);
	auto image2 = cv::imread("F:/mixtrain/label/2.jpg", 1);
	cv::resize(image1, image1, cv::Size(600, 600));
	cv::resize(image2, image2, cv::Size(600, 600));

	auto t1 = LibtorchAsssst::Mat2Tensor(image1);
	auto t2 = LibtorchAsssst::Mat2Tensor(image2);
	

	std::cout << t1.sizes() << std::endl;
	std::cout << t2.sizes() << std::endl;

	std::vector<torch::Tensor>ts;
	ts.push_back(t1);
	ts.push_back(t2);

	torch::TensorList list(ts);

	auto a = torch::cat(list);
	auto b = torch::stack(list);

	std::cout << a.sizes() << std::endl;
	std::cout << b.sizes() << std::endl;

	auto m1 = LibtorchAsssst::Tensor2Mat( b.index({ 0 }));

	cv::imshow("1", m1);

	cv::waitKey();*/

	
	

	return 0;
}