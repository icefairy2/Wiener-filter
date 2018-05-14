#include "stdafx.h"
#include "common.h"
#include <random>

#define WINDOW_SIZE 2

#define NOISE_MEAN 0
#define NOISE_VAR 10


Mat to_uchar_mat(Mat img)
{
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			dst.at<uchar>(i, j) = (uchar)img.at<float>(i, j);
		}
	}
	return dst;
}

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
					stddev += pow(noisy_img.at<uchar>(i, j) - local_mean, 2);
				}
			}
			stddev /= (double)pow(2 * WINDOW_SIZE + 1, 2);
			stddev -= NOISE_VAR;
			return stddev;
		}
	}
	return 0;
}

//Spatial Lee
void simple_wiener(Mat img, Mat noisy_img)
{
	Mat dst_float = Mat(img.rows, img.cols, CV_32FC1);
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	img.copyTo(dst);

	double mean_x;
	double stddev_x;
	double term;

	for (int i = WINDOW_SIZE; i < img.rows - WINDOW_SIZE; i++)
	{
		for (int j = WINDOW_SIZE; j < img.cols - WINDOW_SIZE; j++)
		{
			mean_x = local_mean_4(noisy_img, i, j);
			stddev_x = local_stddev_5(noisy_img, i, j, mean_x);
			term = stddev_x / (stddev_x + NOISE_VAR);
			dst_float.at<float>(i, j) = term * (noisy_img.at<uchar>(i, j) - mean_x) + mean_x;
		}
	}

	dst = to_uchar_mat(dst_float);

	std::cout << "MSE of original image and the simple Wiener filtered: " << mean_squared_error(img, dst) << std::endl;

	imshow("Simple Wiener", dst);
	waitKey(0);
}

int main(int argc, char* argv[])
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat noisy_img = Mat(img.rows, img.cols, CV_8UC1);

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

		std::cout << "MSE of original image and the noisy: " << mean_squared_error(img, noisy_img) << std::endl;

		imshow("Image", img);
		imshow("Noisy", noisy_img);

		simple_wiener(img, noisy_img);
	}
	return 0;
}