#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <istream>
#include <sstream>

#include <yarp/sig/Image.h>
#include <yarp/sig/ImageFile.h>

#include <math.h>
#include <time.h>

#include "NeuronCoders.h"
#include "MyUtils.h"
#include "MyCommon.h"

using namespace yarp::sig;

using namespace std;
 



void robotReferencedPosition(double position[3], double result[2]) {
	double r = 180/MU::PI;		
	result[0] = atan((position[1]-0.6)/position[2]) * r;	// verticalna rovina - otocenie okolo x-ovej osi	
	result[1] = atan(position[0]/position[2]) * r;		// horizontalna rovina - otocenie okolo y-ovej osi	
}


void processImage(string fileName, ostream &outputStream, double scale) {						
	ImageOf<PixelRgb> img;	
	yarp::sig::file::read(img, fileName.c_str());			
	outputStream << endl;		
	for (int y=0; y<img.height(); y++) {
		for (int x=0; x<img.width(); x++) {			
			PixelRgb &pix = img.pixel(x,y);				
			if (x>0) outputStream << " ";					
			double val = scale*(double)pix.r/255.0;
			outputStream << val;					
		}
		outputStream << endl;
	}	
}

bool checkNotBlank(string fileName) {						
	ImageOf<PixelRgb> img;	
	yarp::sig::file::read(img, fileName.c_str());			
	
	int num = 0;
	for (int y=0; y<img.height(); y++) {
		for (int x=0; x<img.width(); x++) {
			PixelRgb &pix = img.pixel(x,y);
			if (pix.r>0) num++;
		}
	}

	return (num>80);
}

void getDimensions(string fileName, double &width, double &height) {
	ImageOf<PixelRgb> img;	
	yarp::sig::file::read(img, fileName.c_str());			
	width = img.width();
	height = img.height();		
}

void copyImage(string oldName, string newName) {						
	ImageOf<PixelRgb> img;	
	yarp::sig::file::read(img, oldName.c_str());			
	yarp::sig::file::write(img, newName.c_str());
}



//see link below to see how data should be organized in file
//num_train_data num_input num_output
//inputdata seperated by space
//outputdata seperated by space
//...
//http://leenissen.dk/fann/html/files/fann_train-h.html#Training_Data_Manipulation


/*
 Nacita subory z generatora a spracuje ich do formatu pre FANN.
 Zaroven vytvori subory pre statistiky.
*/
void processDataForFann(string dir, string name) {
	
	cout << dir << endl;
	
	//file names
		
	string eyesFile = dir +"_eyes.txt";
	string configFile = dir + "_config.txt";
	string objectsFile = dir +"_objects.txt";	
	
	string fileInfo = name + "_info.txt";	
	string fileAll = name + "_all.txt";
	string fileTrain = name + "_train.txt";
	string fileTest = name + "_test.txt";		
	
	string fileAllPositions = name + "_all_positions.txt";
	string fileTrainPositions = name + "_train_positions.txt";
	string fileTestPositions = name + "_test_positions.txt";
	
	string fileAllSizes = name + "_all_sizes.txt";
	string fileTrainSizes = name + "_train_sizes.txt";
	string fileTestSizes = name + "_test_sizes.txt";
	
	string fileAllEyes = name + "_all_eyes.txt";
	string fileTrainEyes = name + "_train_eyes.txt";
	string fileTestEyes = name + "_test_eyes.txt";

	string fileParameters = name + "_parameters.txt";
		
	//load total number of data and image dimensions
	unsigned int 
		totalNum = 0,
		trainNum = 0,
		testNum = 0;
	
	Parameters p;

	ifstream ifs;
	ifs.open(configFile.c_str());
	ifs >> totalNum;
	ifs >> p.retinaWidth;
	ifs >> p.retinaHeight;
	ifs.close();

	double scale = 1.0, imgScale = 1.0;	
	
	
	p.tiltNum = 10+1;										//number of neurons that will code vertical orientation of eye, limits are (-35, 15)
	p.tiltGaussianWidth = 5.0;							    //when neurons are distributed uniformly every 5 degrees, then 2 seem to be reasonable option for gaussian width ("variance")			
	p.tiltGaussianHeight = scale;
	p.tiltPeaks = new double[p.tiltNum];
	NeuronCoders::uniformPeakDistribution(-35, 15, p.tiltNum, p.tiltPeaks);
	
	p.versionNum = 20+1; //(-50,50)
	p.versionGaussianWidth = 7.0;	
	p.versionGaussianHeight = scale, 
	p.versionPeaks = new double[p.versionNum];
	NeuronCoders::uniformPeakDistribution(-50, 50, p.versionNum, p.versionPeaks);

	//const unsigned int 	xNum = 18+1, yNum = 18+1;			//number of coding neurons for horizontal and vertical angle that determine the direction to the object from robot's chest		
	//double xyGaussHeight = scale, xGaussWidth = 5.0, yGaussWidth = 5.0;	

	p.xNum = 2*9+1, 
	p.xGaussianWidth = 10.0, 
	p.xGaussianHeight = scale;
	p.xPeaks = new double[p.xNum];
	NeuronCoders::uniformPeakDistribution(-90, 90, p.xNum, p.xPeaks);

	p.yNum = 2*9+1;
	p.yGaussianWidth = 10.0;	
	p.yGaussianHeight = scale;		
	p.yPeaks = new double[p.yNum];		
	NeuronCoders::uniformPeakDistribution(-90, 90, p.yNum, p.yPeaks);
		
	//save parameters to file
	ofstream ofs;
	ofs.open(fileParameters.c_str());
	printParameters(ofs, p);
	ofs << endl << "--------------------------------" << endl << "This file contains information about parameters that were used for coding angle values to array of neurons for following angles: eye vertical angle (tilt), eye horizontal angle (version), rotation arround x-axis (x-angle), rotation arround y-axis (y-angle). x-angle and y-angle determines object position relative to iCub's chest. Data are in following format: " << endl << "<number_of_neurons_covering_angle_interval> <height_of_gaussian> <width_of_gaussian>" << endl << "<positions_of_peeks...>";
	ofs.close();

	//debug
	cout << "Total: " << totalNum << endl << "Train: " << trainNum << endl << "Test: " << testNum << endl;

	unsigned int 
		numInput = p.tiltNum + p.versionNum + (p.retinaWidth*p.retinaHeight)*2,
		numOutput = p.xNum + p.yNum;			
	
	
	//prepare streams
	ifstream ifsEyes, ifsObjects;

	ifsEyes.open(eyesFile.c_str());
	ifsObjects.open(objectsFile.c_str());
	
	ofstream ofsPositions, ofsSizes, ofsEyes;		
	ofs.open(fileAll.c_str());
	ofsPositions.open(fileAllPositions.c_str());
	ofsSizes.open(fileAllSizes.c_str());
	ofsEyes.open(fileAllEyes.c_str());

	//prepare variables used in loop
	double val = 0.0;			
	ostringstream oss;
	double pos[3], angles[2];
	int index = 0;
	int realnum = 0;
		
	
	//process data - make new files - all, positions, sizes
	for (unsigned int i=0; i<totalNum; i++) {				
		
		/*if (i==trainNum) {
			ofs.close();
			ofs.open(testFile.c_str());
			MU::printlnValues(ofs, "iii", testNum, numInput, numOutput);			
			cout << "Real num "  << realnum << endl;
			realnum = 0;
			cout << "Switching to dataset for testing:" << endl;
		} */		

		bool notblank = true;

		//left eye		
		oss.str("");		
		oss << dir << i << "l.ppm";		
		notblank = checkNotBlank(oss.str());

		//right eye		
		oss.str("");		
		oss << dir << i << "r.ppm";		
		notblank = notblank && checkNotBlank(oss.str());		
		
		if (notblank) {
			realnum++;
		}

		ofs << endl << endl;
		//tilt
		ifsEyes >> val; 	
		if (notblank) NeuronCoders::valueToNeuronArray(val, p.tiltNum, p.tiltPeaks, ofs, p.tiltGaussianHeight, p.tiltGaussianWidth);	
		ofsEyes << val;
		//version
		ifsEyes >> val;				
		if (notblank) NeuronCoders::valueToNeuronArray(val, p.versionNum, p.versionPeaks, ofs, p.versionGaussianHeight, p.versionGaussianWidth);				
		ofsEyes << " " << val << endl;
		
		if (notblank) {
			//left eye		
			oss.str("");		
			oss << dir << i << "l.ppm";		
			processImage(oss.str(), ofs, imgScale);
		
			//right eye		
			oss.str("");		
			oss << dir << i << "r.ppm";		
			processImage(oss.str(), ofs, imgScale);		
		}
	
		int j=0;
		//object position (x,y,z)
		for (j=0; j<3; j++) {
			ifsObjects >> pos[j];
		}				
		robotReferencedPosition(pos, angles);
		if (notblank) {
			MU::printlnArr(ofsPositions, angles, 2);
			ofs << endl;		
			NeuronCoders::valueToNeuronArray(angles[0], p.xNum, p.xPeaks, ofs, p.xGaussianHeight, p.xGaussianWidth);		
			NeuronCoders::valueToNeuronArray(angles[1], p.yNum, p.yPeaks, ofs, p.yGaussianHeight, p.yGaussianWidth);
		}
		
		//skip some data (that are not necessary for now) //shape and size		
		char line[256];
		ifsObjects.getline(line, 256);
		if (notblank) {
			ofsSizes << line << endl;
		}

		/*ifsObjects >> s;					
		j = 3;
		if (s=="sph") { j = 1; } else if (s=="cyl") { j = 2; } else if (s=="box") { j = 3; }				
		while (j > 0) {	
			ifsObjects >> val; 			
			j--;
		}*/							

		//debug
		if ((i+1)%50 == 0) cout << "--processed: " << ( i+1) << endl;
	}		
	cout << "real num" << realnum << endl;
	ofs.close();
	ofs.open(fileInfo.c_str());
	ofs << realnum;

	//close streams
	ofs.close();
	ofsPositions.close();
	ofsSizes.close();	
	ofsEyes.close();
	ifsEyes.close();
	ifsObjects.close();	

	cout << "splitting data to training and testing set" << endl;

	// open read streams
	ifstream ifsPositions, ifsSizes;
	ifs.close();
	ifs.open(fileAll.c_str());
	ifsPositions.open(fileAllPositions.c_str());
	ifsSizes.open(fileAllSizes.c_str());
	ifsEyes.open(fileAllEyes.c_str());
	
	//prepare write streams
	ofs.open(fileTrain.c_str());
	ofsPositions.open(fileTrainPositions.c_str());
	ofsSizes.open(fileTrainSizes.c_str());	
	ofsEyes.open(fileTrainEyes.c_str());
	
	//split all data to train and test set
	trainNum = (realnum * 2) / 3;
	testNum = realnum - trainNum;	
	ofs << trainNum << " " << numInput << " " << numOutput;

	for (int i=0; i<realnum; i++) {
		
		if (i%50==0) cout << i << endl;

		ofs << endl << endl;
		
		double d = 0;
		for (unsigned int j=0; j< p.tiltNum + p.versionNum + (p.retinaWidth*p.retinaHeight)*2 + p.xNum + p.yNum; j++) {
			ifs >> d;
			ofs << d << " ";
			if (j+1==p.tiltNum || j+1==p.tiltNum+p.versionNum || ((j+1-p.tiltNum-p.versionNum)%p.retinaWidth==0) || (j+1==numInput) || (j+1==numInput+p.xNum) ) {
				ofs << endl;	
				if ((j+1==p.tiltNum+p.versionNum) || ((j+1-p.tiltNum-p.versionNum)%(p.retinaWidth*p.retinaHeight)==0)) {
					ofs << endl;
				}
			}			
		}
		ofs << endl;

		char line[255];
		ifsPositions.getline(line, 255);
		ofsPositions << line << endl;
		
		ifsSizes.getline(line, 255);
		ofsSizes << line << endl;	

		ifsEyes.getline(line, 255);
		ofsEyes << line << endl;

		if (i+1==trainNum) {
			ofs.close();
			ofs.open(fileTest.c_str());
			ofs << testNum << " " << numInput << " " << numOutput;

			ofsPositions.close();
			ofsPositions.open(fileTestPositions.c_str());

			ofsSizes.close();
			ofsSizes.open(fileTestSizes.c_str());

			ofsEyes.close();
			ofsEyes.open(fileTestEyes.c_str());
		}
	}

	/*
	ofs.open(fileTrain.c_str());
	MU::printlnValues(ofs, "iii", trainNum, numInput, numOutput);	
	cout << "Processing training dataset: (please wait)" << endl;
	*/	
	
	//close streams	
	ifs.close();
	ifsPositions.close();
	ifsSizes.close();

	ofs.close();
	ofsPositions.close();
	ofsSizes.close();		
	ofsEyes.close();

	//cout << "FANN training file: "  << trainFile << endl << "FANN testing file: " << testFile << endl << "Processing details: " << settingsFile << endl;
}


void repair() {
	string indexFile = "../../data/raw/_indexes.txt";
	string eyesFile = "../../data/raw/_eyes.txt";	
	string objectsFile = "../../data/raw/_objects.txt";	

	string eyesFileR = "../../data/raw/r/_eyes.txt";	
	string objectsFileR = "../../data/raw/r/_objects.txt";	

	ifstream ifsEyes, ifsObjects;
	ofstream oe, oo;

	oe.open(eyesFileR.c_str());
	oo.open(objectsFileR.c_str());
	
	ifsEyes.open(eyesFile.c_str());
	ifsObjects.open(objectsFile.c_str());
	
	ifstream ifsIndex;
	ifsIndex.open(indexFile.c_str());

	int index = 0, j =0, k =0;
	double dump;
	string s;
	int skipped = 0;
	ostringstream o1, o2;

	for (int i=0; i<1434; i++) {		
		ifsIndex >> index;

		o1.str("");		
		o1 << "../../data/raw/" << index << "l.ppm";		
		o2.str("");		
		o2 << "../../data/raw/r/" << i << "l.ppm";		
		copyImage(o1.str(), o2.str());
		
		o1.str("");		
		o1 << "../../data/raw/" << index << "r.ppm";		
		o2.str("");		
		o2 << "../../data/raw/r/" << i << "r.ppm";		
		copyImage(o1.str(), o2.str());

		while (j<index) {
			ifsEyes >> dump >> dump;
			ifsObjects >> dump >> dump >> dump >> s;
			if (s=="sph") { k = 1; } else if (s=="cyl") { k = 2; } else if (s=="box") { k = 3; }
			while (k>0) {
				ifsObjects >> dump;
				k--;
			}
			skipped++;
			j++;
		}

		ifsEyes >> dump; oe << dump;
		ifsEyes >> dump; oe << " " << dump << endl;
		
		ifsObjects >> dump; oo << dump;
		ifsObjects >> dump; oo << " " << dump;
		ifsObjects >> dump; oo << " " << dump;
				
		ifsObjects >> s;					
		oo << " " << s;
		k = 3;
		if (s=="sph") { k = 1; } else if (s=="cyl") { k = 2; } else if (s=="box") { k = 3; }
		while (k>0) { 
			ifsObjects >> dump;
			oo << " " << dump;
			k--;
		}
		oo <<  endl;	

		j++;
	}

	ifsEyes.close();
	ifsObjects.close();
	oo.close();
	oe.close();
	ifsIndex.close();

	cout << skipped;
	MU::pause("preskocenych");

}


int main() {	
	
	string root="", dir="", name="";

	/*cout << "Data processor for FANN: " << endl;
	cout << "Root dir (optional, eg. ../../data/130203-balls/): ";
	cin >> root;
	cout << "Data dir (eg. 1_generator/): ";
	cin >> dir;
	cout << "Output dataset name: (eg. '2_processed/my_dataset'): ";
	cin >> name;
		
	cout << endl << "--processing dataset "  << name << endl;
	*/
	
	root = "../../data/130318-R/";


	dir = "1_generator/";
	name = "2_processed/19";

	processDataForFann(root + dir, root + name);
	
	cout << "DONE."  << endl;
	
	//MU::pause();
	system("pause");

 }