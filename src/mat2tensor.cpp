#include "mat2tensor.h"

torch::Tensor Mat2Tensor(const cv::Mat image)
{
	int channels = image.channels();
	return torch::from_blob(image.data, {channels, image.rows, image.cols }, at::kByte).clone(); 
}