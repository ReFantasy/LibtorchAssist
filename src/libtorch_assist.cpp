//
// Created by TDL on 2020/7/11.
//

#include "libtorch_assist.h"
#include <iostream>

namespace LibtorchAsssst
{
	int Test(int n)
	{
		std::cout << "LibtorchAsssst Test is OK" << std::endl;
		std::cout << "The function input is:" << n << ", return:" << n + 1 << std::endl;
		return n + 1;
	}

}

