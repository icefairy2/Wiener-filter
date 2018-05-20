#include "stdafx.h"
#include "common.h"
#include <random>

#define WINDOW_SIZE 3

#define NOISE_MEAN 0
#define NOISE_VAR 10

#define EPSILON 2.5 * NOISE_VAR
#define WA 0.5

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

//Calculate local mean
double local_mean_9(Mat noisy_img, int row, int col)
{
	if (row - WINDOW_SIZE >= 0 && row + WINDOW_SIZE < noisy_img.rows)
	{
		if (col - WINDOW_SIZE >= 0 && col + WINDOW_SIZE < noisy_img.cols)
		{
			double mean = 0;
			double weight = 0;
			for (int i = row - WINDOW_SIZE; i <= row + WINDOW_SIZE; i++)
			{
				for (int j = col - WINDOW_SIZE; j <= col + WINDOW_SIZE; j++)
				{
					weight = weight_7(noisy_img, row, col, i, j);
					mean += weight * noisy_img.at<uchar>(i, j);
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

//Calculate local standard deviation
double local_stddev_6(Mat noisy_img, int row, int col, double local_mean)
{
	if (row - WINDOW_SIZE >= 0 && row + WINDOW_SIZE < noisy_img.rows)
	{
		if (col - WINDOW_SIZE >= 0 && col + WINDOW_SIZE < noisy_img.cols)
		{
			double stddev = 0;
			double weight = 0;
			for (int i = row - WINDOW_SIZE; i <= row + WINDOW_SIZE; i++)
			{
				for (int j = col - WINDOW_SIZE; j <= col + WINDOW_SIZE; j++)
				{
					weight = weight_7(noisy_img, row, col, i, j);
					stddev += (double)weight * pow(noisy_img.at<uchar>(i, j) - local_mean, 2);
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

//Spatial Lee
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
			stddev_x = local_stddev_6(noisy_img, i, j, mean_x);
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

//Spatial Lee
void awa_wiener(Mat img, Mat noisy_img)
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
			mean_x = local_mean_9(noisy_img, i, j);
			//std::cout << "Mean: " << mean_x << std::endl;
			stddev_x = local_stddev_6(noisy_img, i, j, mean_x);
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

		awa_wiener(img, noisy_img);
	}
	return 0;
}