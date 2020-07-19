#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "libtorch_assist.h"

int main()
{
	auto image = cv::imread("F:/mixtrain/label/1.jpg", 1);
	auto t = LibtorchAsssst::Mat2Tensor(image).to(torch::kFloat);

	auto img = LibtorchAsssst::Tensor2Mat(t);

	cv::imshow("img", img);

	cv::waitKey();
	


	return 0;
}