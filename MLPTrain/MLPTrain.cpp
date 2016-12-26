// MLPTrain.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv\highgui.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgcodecs.hpp"
#include <opencv/cv.h>  
#include <opencv/cvaux.h>  
#include <opencv/highgui.h>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
using namespace ml;

#include <vector>
#include <set>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <opencv2\features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/filesystem.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_features2d310d.lib")
#else
#pragma comment(lib, "opencv_features2d310.lib")
#endif
namespace fs = boost::filesystem;
typedef std::vector<std::string>::const_iterator vec_iter;


const char *defaultNoC[] = { "Ri", "Ko" }; //the names of classes
string mlp_root_dir("E:\\Inv\\MLP"); //
string prjname;
string mlp_trained_data;
string mlp_training_data;
string mlp_test_data;
string mlp_out_dir;
vector<string> NoCs;//names of classes
vector<string> P4training;//classpathes for training
string mlpfile;
string vocafile;
string classfile;
const char objname[] = "opencv_ml_ann_mlp";
const int networkInputSize = 100;
const float trainSplitRatio = 0.75;
int testcount = 100;

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
std::vector<cv::KeyPoint> keypoints;
cv::Mat getDescriptors(const cv::Mat& img)
{
	//cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
	//std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
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
* Read images from a list of file names and returns, for each read image,
* its class name and its local descriptors
*/
void readImages(vec_iter begin, vec_iter end, std::function<void(const std::string&, const cv::Mat&)> callback)
{
	for (auto it = begin; it != end; ++it)
	{
		std::string filename = *it;
		std::cout << "Reading image " << filename << "..." << std::endl;
		cv::Mat img = cv::imread(filename, 0);
		if (img.cols > 600)
			cv::resize(img, img, Size(img.cols / 10, img.rows / 10));
		if (img.empty())
		{
			std::cerr << "WARNING: Could not read image." << std::endl;
			continue;
		}
		std::string classname = getClassName(filename);
		cv::Mat dscp = getDescriptors(img);
		callback(classname, dscp);
	}
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
* Get a trained neural network according to some inputs and outputs
*/
cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples,
	const cv::Mat& trainResponses)
{
	int networkInputSize = trainSamples.cols;
	int networkOutputSize = trainResponses.cols;
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
	std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2,
		networkOutputSize };
	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
	return mlp;
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

/**
* Get a confusion matrix from a set of test samples and their expected
* outputs
*/
typedef std::vector<int> intvec;
std::vector<std::vector<int> > getConfusionMatrix(cv::Ptr<cv::ml::ANN_MLP> mlp,
	const cv::Mat& testSamples, const std::vector<int>& testOutputExpected)
{
	cv::Mat testOutput;
	mlp->predict(testSamples, testOutput);
	std::vector<std::vector<int> > confusionMatrix(2, std::vector<int>(2));
	for (int i = 0; i < testOutput.rows; i++)
	{
		int predictedClass = getPredictedClass(testOutput.row(i));
		int expectedClass = testOutputExpected.at(i);
		confusionMatrix[expectedClass][predictedClass]++;
	}
	return confusionMatrix;
}

void getConfusionMatrix(cv::Ptr<cv::ml::ANN_MLP> mlp,
	const cv::Mat& testSamples,
	const std::vector<int>& testOutputExpected,
	std::vector<intvec>& confusionMatrix)
{
	cv::Mat testOutput;
	mlp->predict(testSamples, testOutput);
	//std::vector<std::vector<int> > confusionMatrix(2, std::vector<int>(2));
	for (int i = 0; i < testOutput.rows; i++)
	{
		int predictedClass = getPredictedClass(testOutput.row(i));
		int expectedClass = testOutputExpected.at(i);
		confusionMatrix[expectedClass][predictedClass]++;
	}
	return;
}
/**
* Print a confusion matrix on screen
*/
void printConfusionMatrix(const std::vector<std::vector<int> >& confusionMatrix,
	const std::set<std::string>& classes)
{
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		std::cout << *it << " ";
	}
	std::cout << std::endl;
	for (size_t i = 0; i < confusionMatrix.size(); i++)
	{
		for (size_t j = 0; j < confusionMatrix[i].size(); j++)
		{
			std::cout << confusionMatrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

/**
* Get the accuracy for a model (i.e., percentage of correctly predicted
* test samples)
*/
float getAccuracy(const std::vector<std::vector<int> >& confusionMatrix)
{
	int hits = 0;
	int total = 0;
	for (size_t i = 0; i < confusionMatrix.size(); i++)
	{
		for (size_t j = 0; j < confusionMatrix.at(i).size(); j++)
		{
			if (i == j) hits += confusionMatrix.at(i).at(j);
			total += confusionMatrix.at(i).at(j);
		}
	}
	return hits / (float)total;
}

/**
* Save our obtained models (neural network, bag of words vocabulary
* and class names) to use it later
*/

void saveModels(string dir,cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary,
	const std::set<std::string>& classes)
{
	mlp->save(mlpfile);
	cv::FileStorage fs(vocafile, cv::FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();
	std::ofstream classesOutput(classfile);
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		classesOutput << getClassId(classes, *it) << "\t" << *it << std::endl;
	}
	classesOutput.close();
}

void consts_init(int argc, _TCHAR* argv[])//(int argc, char** argv)
{
	if (argc < 3)
	{
		NoCs.push_back(string(defaultNoC[0]));
		NoCs.push_back(string(defaultNoC[1]));
	}
	else
	{

		NoCs.push_back(string(argv[1]));
		NoCs.push_back(string(argv[2]));
	}
	prjname = NoCs[0] + NoCs[1];
	mlp_trained_data = string(mlp_root_dir) + string("\\TrainedData\\") + prjname;
	mlp_training_data = string(mlp_root_dir) + string("\\TrainingData\\") + prjname;
	mlp_test_data = string(mlp_root_dir) + string("\\TestData\\") + prjname;
	mlp_out_dir = string(mlp_root_dir) + string("\\OutData\\") + prjname;
	for (auto it = NoCs.begin(); it != NoCs.end(); ++it)
	{
		P4training.push_back(mlp_training_data + string("\\") + *it);
	}
	mlpfile = mlp_trained_data + string("\\") + string("mlp.yaml");
	vocafile = mlp_trained_data + string("\\") + string("vocabulary.yaml");
	classfile = mlp_trained_data + string("\\") + string("classes.txt");
}
int _tmain(int argc, _TCHAR* argv[])//int main(int argc, char** argv)//
{
	//
	consts_init(argc, argv);
	//
	std::cout << "Reading training set..." << std::endl;
	double start = (double)cv::getTickCount();
	std::vector<std::string> dogs = getFilesInDirectory(P4training[0]);
	std::vector<std::string> cats = getFilesInDirectory(P4training[1]);
	std::random_shuffle(dogs.begin(), dogs.end());
	std::random_shuffle(cats.begin(), cats.end());

	cv::Mat descriptorsSet;
	std::vector<ImageData*> descriptorsMetadata;
	std::set<std::string> classes;

	readImages(dogs.begin(), dogs.begin() + (size_t)(dogs.size() * trainSplitRatio),
		[&](const std::string& classname, const cv::Mat& descriptors) {
		// Append to the set of classes
		classes.insert(classname);
		// Append to the list of descriptors
		descriptorsSet.push_back(descriptors);
		// Append metadata to each extracted feature
		ImageData* data = new ImageData;
		data->classname = classname;
		data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
		for (int j = 0; j < descriptors.rows; j++)
		{
			descriptorsMetadata.push_back(data);
		}
	});
	readImages(cats.begin(), cats.begin() + (size_t)(cats.size() * trainSplitRatio),
		[&](const std::string& classname, const cv::Mat& descriptors) {
		// Append to the set of classes
		classes.insert(classname);
		// Append to the list of descriptors
		descriptorsSet.push_back(descriptors);
		// Append metadata to each extracted feature
		ImageData* data = new ImageData;
		data->classname = classname;
		data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
		for (int j = 0; j < descriptors.rows; j++)
		{
			descriptorsMetadata.push_back(data);
		}
	});

	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	std::cout << "Creating vocabulary..." << std::endl;
	start = (double)cv::getTickCount();
	cv::Mat labels;
	cv::Mat vocabulary;
	// Use k-means to find k centroids (the words of our vocabulary)
	cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
		cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
	// No need to keep it on memory anymore
	descriptorsSet.release();
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	// Convert a set of local features for each image in a single descriptors
	// using the bag of words technique
	std::cout << "Getting histograms of visual words..." << std::endl;
	int* ptrLabels = (int*)(labels.data);
	int size = labels.rows * labels.cols;
	for (int i = 0; i < size; i++)
	{
		int label = *ptrLabels++;
		ImageData* data = descriptorsMetadata[i];
		data->bowFeatures.at<float>(label)++;
	}
	// Filling matrixes to be used by the neural network
	std::cout << "Preparing neural network..." << std::endl;
	cv::Mat trainSamples;
	cv::Mat trainResponses;
	std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
	for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end();)
	{
		ImageData* data = *it;
		cv::Mat normalizedHist;
		cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
		trainSamples.push_back(normalizedHist);
		trainResponses.push_back(getClassCode(classes, data->classname));
		delete *it; // clear memory
		it++;
	}
	descriptorsMetadata.clear();
	// Training neural network
	std::cout << "Training neural network..." << std::endl;
	start = (double)cv::getTickCount();
	cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	// We can clear memory now 
	trainSamples.release();
	trainResponses.release();

	// Train FLANN 
	std::cout << "Training FLANN..." << std::endl;
	start = (double)cv::getTickCount();
	cv::FlannBasedMatcher flann;
	flann.add(vocabulary);
	flann.train();
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	// Reading test set 
	std::cout << "Reading test set..." << std::endl;
	start = (double)cv::getTickCount();
	cv::Mat testSamples;
	std::vector<int> testOutputExpected;

	readImages(dogs.begin() + (size_t)(dogs.size() * trainSplitRatio), dogs.end(),
		[&](const std::string& classname, const cv::Mat& descriptors) {
		// Get histogram of visual words using bag of words technique
		cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
		cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
		testSamples.push_back(bowFeatures);
		testOutputExpected.push_back(getClassId(classes, classname));
	});
	readImages(cats.begin() + (size_t)(cats.size() * trainSplitRatio), cats.end(),
		[&](const std::string& classname, const cv::Mat& descriptors) {
		// Get histogram of visual words using bag of words technique
		cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
		cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
		testSamples.push_back(bowFeatures);
		testOutputExpected.push_back(getClassId(classes, classname));
	});
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	// Get confusion matrix of the test set
	std::vector<std::vector<int> > confusionMatrix = getConfusionMatrix(mlp,
																		testSamples,
																		testOutputExpected);
	// Get accuracy of our model
	std::cout << "Confusion matrix: " << std::endl;
	printConfusionMatrix(confusionMatrix, classes);
	std::cout << "Accuracy: " << getAccuracy(confusionMatrix) << std::endl;
	// Save models
	std::cout << "Saving models..." << std::endl;
	saveModels(mlp_trained_data, mlp, vocabulary, classes);

	return 0;
}


