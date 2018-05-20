#include "stdafx.h"
#include "common.h"
#include <random>

#define WINDOW_SIZE 3

#define NOISE_MEAN 0
#define NOISE_VAR 10

#define EPSILON 2.5 * NOISE_VAR
#define WA 0.1

// Calculate MSE, the mean squared error
double mean_squared_error(Mat orig_img, Mat result_img)
{
	double mse = 0;
	const int N = orig_img.rows * orig_img.cols;
	for (int i = 0; i < orig_img.rows; i++)
	{
		for (int j = 0; j < orig_img.cols; j++)
		{
			mse += pow(result_img.at<uchar>(i, j) - orig_img.at<uchar>(i, j), 2);
		}
	}
	mse /= (float)N;
	return mse;
}

//Gaussian blur
Mat gaussian_2d_filter(Mat img, Mat noisy_img)
{
	Mat dst = Mat(noisy_img.rows, noisy_img.cols, CV_8UC1);

	img.copyTo(dst);

	int i, j, x, y;
	float g[200][200];
	int w = WINDOW_SIZE;//2 * WINDOW_SIZE + 1;

	float sigma = (float)w / 6;
	float term = 1 / (2 * PI *pow(sigma, 2));

	for (x = 0; x < w; x++)
	{
		for (y = 0; y < w; y++)
		{
			g[x][y] = term * exp(-(pow(x - w / 2, 2) + pow(y - w / 2, 2)) / (2 * pow(sigma, 2)));
		}
	}

	float conv;

	for (i = w / 2; i < noisy_img.rows - w / 2; i++)
	{
		for (j = w / 2; j < noisy_img.cols - w / 2; j++)
		{
			conv = 0;
			for (x = 0; x < w; x++)
			{
				for (y = 0; y < w; y++)
				{
					conv += g[x][y] * (float)noisy_img.at<uchar>(i + x - w / 2, j + y - w / 2);
				}
			}
			dst.at<uchar>(i, j) = conv;
		}
	}

	std::cout << "MSE of original image and the Gauss blurred: " << mean_squared_error(img, dst) << std::endl;

	return dst;
}

//Calculate weight
double weight_7(Mat noisy_img, int i, int j, int p, int q)
{
	if (i == p && j == q)
	{
		return 0;
	}
	//Calculate K normalization constant
	double k = 0;
	double fraction = 0;
	for (int n = i - WINDOW_SIZE; n <= i + WINDOW_SIZE; n++)
	{
		for (int m = j - WINDOW_SIZE; m <= j + WINDOW_SIZE; m++)
		{
			fraction = (double)max(pow(EPSILON, 2), pow(noisy_img.at<uchar>(i, j) - noisy_img.at<uchar>(n, m), 2));
			fraction = 1 / (1 + WA * fraction);
			k += fraction;
		}
	}
	k = 1 / k;

	//Calculate the weight
	double weight = 0;
	weight = (double)max(pow(EPSILON, 2), pow(noisy_img.at<uchar>(i, j) - noisy_img.at<uchar>(p, q), 2));
	weight = k / (1 + WA * weight);
	return weight;
}


//Calculate local mean
double local_mean_4(Mat noisy_img, int row, int col)
{
	if (row - WINDOW_SIZE >= 0 && row + WINDOW_SIZE < noisy_img.rows)
	{
		if (col - WINDOW_SIZE >= 0 && col + WINDOW_SIZE < noisy_img.cols)
		{
			double mean = 0;
			for (int i = row - WINDOW_SIZE; i <= row + WINDOW_SIZE; i++)
			{
				for (int j = col - WINDOW_SIZE; j <= col + WINDOW_SIZE; j++)
				{
					mean += noisy_img.at<uchar>(i, j);
				}
			}
			mean /= (double)pow(2 * WINDOW_SIZE + 1, 2);
			return mean;
		}
	}
	return 0;
}

//Calculate local standard deviation
double local_stddev_5(Mat noisy_img, int row, int col, double local_mean)
{
	if (row - WINDOW_SIZE >= 0 && row + WINDOW_SIZE < noisy_img.rows)
	{
		if (col - WINDOW_SIZE >= 0 && col + WINDOW_SIZE < noisy_img.cols)
		{
			double stddev = 0;
			for (int i = row - WINDOW_SIZE; i <= row + WINDOW_SIZE; i++)
			{
				for (int j = col - WINDOW_SIZE; j <= col + WINDOW_SIZE; j++)
				{
					stddev += (double)pow(noisy_img.at<uchar>(i, j) - local_mean, 2);
				}
			}
			stddev /= (double)pow(2 * WINDOW_SIZE + 1, 2);
			stddev -= NOISE_VAR * NOISE_VAR;
			return stddev;
		}
	}
	return 0;
}

//Spatial Lee
void simple_wiener(Mat img, Mat noisy_img)
{
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	img.copyTo(dst);

	double mean_x;
	double stddev_x;
	double term;

	double var;

	for (int i = WINDOW_SIZE; i < noisy_img.rows - WINDOW_SIZE; i++)
	{
		for (int j = WINDOW_SIZE; j < noisy_img.cols - WINDOW_SIZE; j++)
		{
			mean_x = local_mean_4(noisy_img, i, j);
			//std::cout << "Mean: " << mean_x << std::endl;
			stddev_x = local_stddev_5(noisy_img, i, j, mean_x);
			//std::cout << "Standard Dev: " << stddev_x << std::endl;
			term = stddev_x / (stddev_x + NOISE_VAR * NOISE_VAR);
			//std::cout << "Term: " << term << std::endl;
			var = term * (noisy_img.at<uchar>(i, j) - mean_x) + mean_x;
			//std::cout << "Var: " << var << "  " << (int)var << std::endl;
			dst.at<uchar>(i, j) = (int)var;
		}
	}

	std::cout << "MSE of original image and the simple Wiener filtered: " << mean_squared_error(img, dst) << std::endl;

	imshow("Simple Wiener", dst);
	waitKey(0);
}

//Spatial Kuan AWA
void kuan_wiener(Mat img, Mat noisy_img)
{
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	img.copyTo(dst);

	double mean_x;
	double stddev_x;
	double term;

	double var;

	for (int i = WINDOW_SIZE; i < noisy_img.rows - WINDOW_SIZE; i++)
	{
		for (int j = WINDOW_SIZE; j < noisy_img.cols - WINDOW_SIZE; j++)
		{
			mean_x = local_mean_4(noisy_img, i, j);
			//std::cout << "Mean: " << mean_x << std::endl;					
			double weight = 0;
			stddev_x = 0;
			for (int p = i - WINDOW_SIZE; p <= i + WINDOW_SIZE; p++)
			{
				for (int q = j - WINDOW_SIZE; q <= j + WINDOW_SIZE; q++)
				{
					weight = weight_7(noisy_img, i, j, p, q);
					stddev_x += (double)weight * pow(noisy_img.at<uchar>(p, q) - mean_x, 2);
				}
			}
			//std::cout << "Standard Dev: " << stddev_x << std::endl;
			term = stddev_x / (stddev_x + NOISE_VAR * NOISE_VAR);
			//std::cout << "Term: " << term << std::endl;
			var = term * (noisy_img.at<uchar>(i, j) - mean_x) + mean_x;
			//std::cout << "Var: " << var << "  " << (int)var << std::endl;
			dst.at<uchar>(i, j) = (int)var;
		}
	}

	std::cout << "MSE of original image and the Kuan Wiener filtered: " << mean_squared_error(img, dst) << std::endl;

	imshow("Kuan Wiener", dst);
	waitKey(0);
}

//Spatial AWA
void awa_wiener(Mat img, Mat noisy_img)
{
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	Mat means = Mat::zeros(img.rows, img.cols, CV_32FC1);
	img.copyTo(dst);

	double mean_x;
	double stddev_x;
	double term;

	double var;
	double mean;
	double weight;

	//calculate weights
	for (int i = WINDOW_SIZE; i < noisy_img.rows - WINDOW_SIZE; i++)
	{
		for (int j = WINDOW_SIZE; j < noisy_img.cols - WINDOW_SIZE; j++)
		{
			mean = 0;
			for (int p = i - WINDOW_SIZE; p <= i + WINDOW_SIZE; p++)
			{
				for (int q = j - WINDOW_SIZE; q <= j + WINDOW_SIZE; q++)
				{
					weight = weight_7(noisy_img, i, j, p, q);
					mean += weight * noisy_img.at<uchar>(p, q);
				}
			}
			means.at<float>(i, j) = mean;
		}
	}

	for (int i = WINDOW_SIZE; i < noisy_img.rows - WINDOW_SIZE; i++)
	{
		for (int j = WINDOW_SIZE; j < noisy_img.cols - WINDOW_SIZE; j++)
		{
			mean_x = means.at<float>(i, j);
			//std::cout << "Mean: " << mean_x << std::endl;
			stddev_x = 0;
			for (int p = i - WINDOW_SIZE; p <= i + WINDOW_SIZE; p++)
			{
				for (int q = j - WINDOW_SIZE; q <= j + WINDOW_SIZE; q++)
				{
					weight = weight_7(noisy_img, i, j, p, q);
					stddev_x += weight * pow(noisy_img.at<uchar>(p, q) - mean_x, 2);
				}
			}
			//std::cout << "Standard Dev: " << stddev_x << std::endl;
			term = stddev_x / (stddev_x + NOISE_VAR * NOISE_VAR);
			//std::cout << "Term: " << term << std::endl;
			var = term * (noisy_img.at<uchar>(i, j) - mean_x) + mean_x;
			//std::cout << "Var: " << var << "  " << (int)var << std::endl;
			dst.at<uchar>(i, j) = (int)var;
		}
	}

	std::cout << "MSE of original image and the awa Wiener filtered: " << mean_squared_error(img, dst) << std::endl;

	imshow("AWA Wiener", dst);
	waitKey(0);
}


int main(int argc, char* argv[])
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat noisy_img = Mat(img.rows, img.cols, CV_8UC1);
		Mat gauss_blur;

		// Define random generator with Gaussian distribution
		std::default_random_engine generator;
		std::normal_distribution<double> dist(NOISE_MEAN, NOISE_VAR);

		int temp;

		// Add Gaussian noise
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++)
			{
				temp = (int)img.at<uchar>(i, j) + dist(generator);
				if (temp > 255)
				{
					temp = 255;
				}
				if (temp < 0)
				{
					temp = 0;
				}
				noisy_img.at<uchar>(i, j) = temp;
			}

		}

		gauss_blur = gaussian_2d_filter(img, noisy_img);

		std::cout << std::endl;

		std::cout << "Choose your filtering method:" << std::endl;
		std::cout << "1 - Spatial Lee - simple Wiener filter" << std::endl;
		std::cout << "2 - Spatial Kuan AWA - only the standard deviation is weighted" << std::endl;
		std::cout << "3 - Spatial AWA - adaptive Wiener filter" << std::endl;

		int method;
		std::cin >> method;
		switch (method)
		{
		case 1:
			imshow("Image", img);
			imshow("Noisy", noisy_img);
			imshow("Gauss blur", gauss_blur);
			std::cout << "MSE of original image and the noisy: " << mean_squared_error(img, noisy_img) << std::endl;
			simple_wiener(img, noisy_img);
		case 2:
			imshow("Image", img);
			imshow("Noisy", noisy_img);
			imshow("Gauss blur", gauss_blur);
			std::cout << "MSE of original image and the noisy: " << mean_squared_error(img, noisy_img) << std::endl;

			kuan_wiener(img, noisy_img);
		case 3:
			imshow("Image", img);
			imshow("Noisy", noisy_img);
			imshow("Gauss blur", gauss_blur);
			std::cout << "MSE of original image and the noisy: " << mean_squared_error(img, noisy_img) << std::endl;
			awa_wiener(img, noisy_img);
		default:
			imshow("Image", img);
			imshow("Noisy", noisy_img);
			imshow("Gauss blur", gauss_blur);
			std::cout << "MSE of original image and the noisy: " << mean_squared_error(img, noisy_img) << std::endl;
			std::cout << "Wrong input!" << std::endl;
		}
	}
	return 0;
}