#ifndef __MAT2TENSOR_H__
#define __MAT2TENSOR_H__
#include "opencv2/opencv.hpp"
#include "torch/torch.h"

torch::Tensor Mat2Tensor(const cv::Mat image);

#endif//__MAT2TENSOR_H__
