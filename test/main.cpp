#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "libtorch_assist.h"


class Dataset :public torch::data::Dataset<Dataset>
{
public:
	Dataset(std::filesystem::path input_path, std::filesystem::path labels_path)
	{
		data = LibtorchAsssst::ReadImagesFromFolder(input_path, 1, { 256,256 }).to(torch::kFloat);
		data.div_(255);
		labels = LibtorchAsssst::ReadImagesFromFolder(labels_path, 1, { 256,256 }).to(torch::kFloat);
		labels.div_(255);
	}

	torch::data::Example<> get(size_t index)override
	{
		return { data[index], labels[index] };
	}

	torch::optional<size_t> size()const override
	{
		return labels.size(0);
	}

private:
	torch::Tensor data;
	torch::Tensor labels;
};

struct Net :public torch::nn::Module
{
	Net()
		:conv1(torch::nn::Conv2dOptions(3, 6, 3).padding(1)),
		conv2(torch::nn::Conv2dOptions(6, 8, 3).padding(1)),
		conv3(torch::nn::Conv2dOptions(8, 16, 3).padding(1)),
		conv4(torch::nn::Conv2dOptions(16, 32, 3).padding(1)),
		conv5(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
		conv6(torch::nn::Conv2dOptions(64, 32, 3).padding(1)),
		conv7(torch::nn::Conv2dOptions(32, 8, 3).padding(1)),
		conv8(torch::nn::Conv2dOptions(8, 6, 3).padding(1)),
		conv9(torch::nn::Conv2dOptions(6, 3, 3).padding(1))
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);
		register_module("conv5", conv5);
		register_module("conv6", conv6);
		register_module("conv7", conv7);
		register_module("conv8", conv8);
		register_module("conv9", conv9);
	}

		torch::Tensor forward(torch::Tensor x)
	{

		x = torch::relu(conv1(x));

		x = torch::relu(conv2(x));

		x = torch::relu(conv3(x));

		x = torch::relu(conv4(x));

		x = torch::relu(conv5(x));

		x = torch::relu(conv6(x));

		x = torch::relu(conv7(x));

		x = torch::relu(conv8(x));

		x = torch::relu(conv9(x));

		return x;
	}



		torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr } , conv4{ nullptr }, conv5{ nullptr }, conv6{ nullptr }, conv7{ nullptr }, conv8{ nullptr }, conv9{ nullptr };
};


using namespace std;
int main()
{
	int kEpoch = 20000;
	auto device = torch::Device(torch::kCUDA);
	int batch_size = 5;

	
	auto data_root_dir = std::filesystem::current_path().parent_path();
	auto input_path = (data_root_dir / "mixtrain/input").make_preferred();
	auto labels_path = (data_root_dir / "mixtrain/label").make_preferred();
	std::cout << input_path << std::endl;
	std::cout << labels_path << std::endl;

	
	auto src_dataset = Dataset(input_path, labels_path);
	auto dataset = src_dataset.map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(batch_size));

	Net net;
	net.to(device);


	//torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(0.01));
	torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(2e-4).betas(std::tuple<double, double>{ 0.5, 0.5 }));

	net.train();
	for (int i = 0; i < kEpoch; i++)
	{
		for (auto& batch : *data_loader)
		{
			auto data = batch.data.to(device);

			auto target = batch.target.to(device);

			optimizer.zero_grad();

			auto output = net.forward(data);
			auto loss = torch::mse_loss(output, target);

			loss.backward();

			optimizer.step();

			printf("%5.5f\n", loss.item<float>());


		}
		std::cout << "epoch: " << i << std::endl;
	}

	
	auto _data = src_dataset.get(0).data;
	auto src = LibtorchAsssst::Tensor2Mat(_data*255);

	net.eval();
	auto x = _data.unsqueeze(0);
	x = x.to(torch::kFloat).to(device);
	auto y = net.forward(x).squeeze().to(torch::kCPU);
	auto dst = LibtorchAsssst::Tensor2Mat(y*256);

	cv::imshow("s", src);
	cv::imshow("p", dst);


	cv::waitKey();
	return 0;
}