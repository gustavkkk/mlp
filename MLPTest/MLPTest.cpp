// MLPTest.cpp : Defines the entry point for the console application.
//

#include <stdafx.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/filesystem.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_features2d310d.lib")
#else
#pragma comment(lib, "opencv_features2d310.lib")
#endif

using namespace cv;
using namespace std;
using namespace ml;

namespace fs = boost::filesystem;

typedef std::vector<std::string>::const_iterator vec_iter;

struct ImageData
{
	std::string classname;
	cv::Mat bowFeatures;
};

/**
* Get all files in directory (not recursive)
* @param directory Directory where the files are contained
* @return A list containing the file name of all files inside given directory
**/
std::vector<std::string> getFilesInDirectory(const std::string& directory)
{
	std::vector<std::string> files;
	fs::path root(directory);
	fs::directory_iterator it_end;
	for (fs::directory_iterator it(root); it != it_end; ++it)
	{
		if (fs::is_regular_file(it->path()))
		{
			files.push_back(it->path().string());
		}
	}
	return files;
}

/**
* Extract the class name from a file name
*/
inline std::string getClassName(const std::string& filename)
{
	//return filename.substr(filename.find_last_of('/') + 1, 3);
	int pos = filename.find_last_of("\\");
	return filename.substr(pos - 4, 3);
}

/**
* Extract local features for an image
*/
cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
//Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");//BRIEF,SURF,SIFT
std::vector<cv::KeyPoint> keypoints;
cv::Mat getDescriptors(const cv::Mat& img)
{
	//cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
	//std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	//Ptr<FeatureDetector> detector(new DynamicAdaptedFeatureDetector(new FastAdjuster(30, true), 600, 1500, 10));//StarAdjuster(),FastAdjuster(30,true), SurfAdjuster()
	//kaze->detect(img, keypoints);
	kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
	keypoints.clear();
	return descriptors;
}
void getDescriptors(const cv::Mat img, Mat &desp)
{
	//cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
	//std::vector<cv::KeyPoint> keypoints;
	kaze->detectAndCompute(img, cv::noArray(), keypoints, desp);
	keypoints.clear();
	return;
}
/**
* Transform a class name into an id
*/
int getClassId(const std::set<std::string>& classes, const std::string& classname)
{
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		if (*it == classname) break;
		++index;
	}
	return index;
}

/**
* Get a binary code associated to a class
*/
cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname)
{
	cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
	int index = getClassId(classes, classname);
	code.at<float>(index) = 1;
	return code;
}

/**
* Turn local features into a single bag of words histogram of
* of visual words (a.k.a., bag of words features)
*/
std::vector<cv::DMatch> matches;
cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
	int vocabularySize)
{
	cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
	//std::vector<cv::DMatch> matches;
	flann.match(descriptors, matches);
	for (size_t j = 0; j < matches.size(); j++)
	{
		int visualWord = matches[j].trainIdx;
		outputArray.at<float>(visualWord)++;
	}
	matches.clear();
	return outputArray;
}

void getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
	int vocabularySize, cv::Mat &outputArray)
{
	outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
	//std::vector<cv::DMatch> matches;
	flann.match(descriptors, matches);
	for (size_t j = 0; j < matches.size(); j++)
	{
		int visualWord = matches[j].trainIdx;
		outputArray.at<float>(visualWord)++;
	}
	matches.clear();
	return;
}

/**
* Receives a column matrix contained the probabilities associated to
* each class and returns the id of column which contains the highest
* probability
*/
int getPredictedClass(const cv::Mat& predictions)
{
	float maxPrediction = predictions.at<float>(0);
	float maxPredictionIndex = 0;
	const float* ptrPredictions = predictions.ptr<float>(0);
	for (int i = 0; i < predictions.cols; i++)
	{
		float prediction = *ptrPredictions++;
		if (prediction > maxPrediction)
		{
			maxPrediction = prediction;
			maxPredictionIndex = (float)i;
		}
	}
	return (int)maxPredictionIndex;
}

const char *defaultNoC[] = { "Ri", "Ko" }; //the names of classes
string mlp_root_dir("E:\\Inv\\MLP"); //
string prjname;
string mlp_trained_data;
string mlp_training_data;
string mlp_test_data;
string mlp_out_dir;
vector<string> NoCs;//names of classes
string mlpfile;
string vocafile;
string classfile;
const char objname[] = "opencv_ml_ann_mlp";
const int networkInputSize = 100;
const float trainSplitRatio = 0.75;
int testcount = 100;

void consts_init(int argc, char** argv)
{
	if (argc < 4)
	{
		NoCs.push_back(string(defaultNoC[0]));
		NoCs.push_back(string(defaultNoC[1]));
	}
	else
	{
		NoCs.push_back(string(argv[1]));
		NoCs.push_back(string(argv[2]));
		testcount = atoi(argv[3]);
	}
	prjname = NoCs[0] + NoCs[1];
	mlp_trained_data = string(mlp_root_dir) + string("\\TrainedData\\") + prjname;
	mlp_test_data = string(mlp_root_dir) + string("\\TestData\\") + prjname;
	mlp_out_dir = string(mlp_root_dir) + string("\\OutData\\") + prjname;
	mlpfile = mlp_trained_data + string("\\") + string("mlp.yaml");
	vocafile = mlp_trained_data + string("\\") + string("vocabulary.yaml");
	classfile = mlp_trained_data + string("\\") + string("classes.txt");
}

void mlp_init(cv::Ptr<cv::ml::ANN_MLP> mlp_t, cv::FlannBasedMatcher &flann_t)
{
	Mat voca;
	//
	FileStorage fs(mlpfile.c_str(), FileStorage::READ);
	FileNode fn = fs[objname];
	mlp_t->read(fn);
	//
	FileStorage fs2(vocafile.c_str(), FileStorage::READ);
	fs2["vocabulary"] >> voca;
	fs2.release();
	//
	flann_t.add(voca);
	flann_t.train();
}

int mlp_predict(Mat img, cv::Ptr<cv::ml::ANN_MLP> mlp_t, cv::FlannBasedMatcher &flann_t, int size)
{
	cv::Mat dscp = getDescriptors(img);
	cv::Mat bowFeatures = getBOWFeatures(flann_t, dscp, size);
	cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::Mat samples;
	samples.push_back(bowFeatures);
	cv::Mat output;
	mlp_t->predict(samples, output);
	int predictedClass = getPredictedClass(output.row(0));
	return predictedClass;
}

void showResult(string isWhat, Mat img, int predictedValue)
{
	imshow(NoCs[predictedValue], img);
	waitKey(1000);
}

void discriminate(string filename, Mat img, int predictedValue)
{
	std::cout << "This is " << NoCs[predictedValue] << "." << std::endl;
	string fullpath = mlp_out_dir +
		string("\\") +
		NoCs[predictedValue] +
		string("\\") +
		filename +
		string(".jpg");
	imwrite(fullpath, img);
	return;
}

int _tmain(int argc, char* argv[])//int main(int argc, char** argv)
{
	cv::Ptr<cv::ml::ANN_MLP> mlp_t = cv::ml::ANN_MLP::create();
	cv::FlannBasedMatcher flann_t;
	//
	consts_init(argc, argv);
	//
	mlp_init(mlp_t,flann_t);
	//
	std::vector<std::string> testdata = getFilesInDirectory(mlp_test_data);
	std::random_shuffle(testdata.begin(), testdata.end());	
	auto it = testdata.begin();
	for (int i = 0; i < testcount; i++)//auto it = testdata.begin(); it != (testdata.begin() + testcount); ++it)
	{
		std::string fullpath = *it;
		cv::Mat img = cv::imread(fullpath.c_str(), 0);
		if (img.cols > 600)
			cv::resize(img, img, Size(img.cols / 10, img.rows / 10));
		cv::Mat clrimg = cv::imread(fullpath.c_str(), 1);
		int predictedClass = mlp_predict(img, mlp_t, flann_t, networkInputSize);	
		discriminate(std::to_string(i),
			clrimg,
			predictedClass);
		++it;
		if (it == testdata.end())
			break;
	}
	return 0;
}

