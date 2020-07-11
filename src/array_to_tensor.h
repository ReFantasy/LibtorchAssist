//
// Created by TDL on 2020/7/11.
//

#ifndef LIBTORCHASSIST_ARRAY_TO_TENSOR_H
#define LIBTORCHASSIST_ARRAY_TO_TENSOR_H

#include <torch/torch.h>

namespace LibtorchAsssst
{
	/**
	 * C++一维数组转 torch::Tensor
	 * @tparam N 数组大小
	 * @param array
	 * @return
	 */
	template <size_t N>
	torch::Tensor Array2Tensor(float (&array)[N])
	{
		auto tensor = torch::from_blob(array, { N }, torch::kFloat);
		return tensor.clone();
	}
	template <size_t N>
	torch::Tensor Array2Tensor(float (&&array)[N])
	{
		return Array2Tensor(array);
	}


	/**
	 * C++二维数组转 torch::Tensor
	 * @tparam M 数组行数
	 * @tparam N 数组列数
	 * @param array2d 二维数组
	 * @return torch::Tensor
	 */
	template<size_t M, size_t N>
	torch::Tensor Array2Tensor(float (& array2d)[M][N])
	{
		auto tensor = torch::from_blob(array2d, { M, N }, torch::kFloat);
		return tensor.clone();
	}

	template<size_t M, size_t N>
	torch::Tensor Array2Tensor(float (&& array2d)[M][N])
	{
		return Array2Tensor(array2d);
	}


	/**
	 * C++三维数组转 torch::Tensor
	 * @tparam C 通道数
	 * @tparam M 数组行数
	 * @tparam N 数组列数
	 * @param array3d 三维数组
	 * @return torch::Tensor
	 */
	template<size_t C, size_t M, size_t N>
	torch::Tensor Array2Tensor(float (& array3d)[C][M][N])
	{
		auto tensor = torch::from_blob(array3d, { C, M, N }, torch::kFloat);
		return tensor.clone();
	}

	template<size_t C, size_t M, size_t N>
	torch::Tensor Array2Tensor(float (&& array3d)[C][M][N])
	{
		return Array2Tensor(array3d);
	}

	/**
	 * C++四维数组转 torch::Tensor
	 * @tparam K 样本数
	 * @tparam C 通道数
	 * @tparam M 数组行数
	 * @tparam N 数组列数
	 * @param array4d 四维数组
	 * @return torch::Tensor
	 */
	template<size_t K, size_t C, size_t M, size_t N>
	torch::Tensor Array2Tensor(float (& array4d)[K][C][M][N])
	{
		auto tensor = torch::from_blob(array4d, { K, C, M, N }, torch::kFloat);
		return tensor.clone();
	}

	template<size_t K, size_t C, size_t M, size_t N>
	torch::Tensor Array2Tensor(float (&& array4d)[K][C][M][N])
	{
		return Array2Tensor(array4d);
	}


}

#endif //LIBTORCHASSIST_ARRAY_TO_TENSOR_H
