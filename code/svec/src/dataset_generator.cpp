#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include <yarp/os/Network.h>


#include <yarp/sig/Image.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Vector.h>
#include <yarp/os/RpcClient.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IPositionControl.h>
#include <yarp/dev/IEncoders.h>
#include <yarp/dev/IVelocityControl.h>
#include <yarp/dev/ControlBoardInterfaces.h>

/*#include "opencv/cv.h"
#include "opencv/cvaux.h"
#include "opencv/highgui.h"*/

#include <yarp/sig/ImageFile.h>

#include <math.h>
#include <time.h>
#include <Windows.h>

#include "MyUtils.h"
#include "MySim.h"




/* Get all OS and signal processing YARP classes */

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;
using namespace std;
 
  
/*
void modulationArray(double value, double min, double max, double numberOfNeurons, double *result) {
	double step = ((max - min)+1)/numberOfNeurons;
	for (int i=0; i<numberOfNeurons; i++) {
		double x = min + (double)i*step;
		result[i] = MU::gaussian(value, 1, x, 1);
	}
}

void robotReferencedPosition(double x, double y, double z, double *result) {	
	double r = 180/MU::PI;		
	result[0] = atan(x/z) * r;		// horizontalna rovina - otocenie okolo y-ovej osi
	result[1] = atan((y-0.6)/z) * r;	// verticalna rovina - otocenie okolo x-ovej osi	
}
*/

void dummyInitialization() {
	if (!Network::exists("/icubSim/world")) {
		system("start cmd /K icub_SIM");		
	}	
	int r_lefteye = 0;
	if (!Network::exists("/lefteye")) {
		r_lefteye = system("start cmd /c yarpview /lefteye");
	}
	int r_righteye = 0;
	if (!Network::exists("/righteye")) {
		r_righteye = system("start cmd /c yarpview /righteye");
	}	
	Sleep(2000);
	if ((r_lefteye == 0) && !Network::isConnected("/icubSim/cam/left", "/lefteye")) Network::connect("/icubSim/cam/left", "/lefteye");
	if ((r_righteye == 0) && !Network::isConnected("/icubSim/cam/right", "/righteye")) Network::connect("/icubSim/cam/right", "/righteye");	
}


bool checkNotBlank(ImageOf<PixelRgb> img, double ratio = 0.012) {
	int num = 0;
	for (int y=0; y<img.height(); y++) {
		for (int x=0; x<img.width(); x++) {
			PixelRgb &pix = img.pixel(x,y);
			if (pix.r>0) num++;
		}
	}
	return (( (double) num) / ( (double) (img.height() * img.width() ))) > ratio;	
}

void processImage(ImageOf<PixelRgb> *image, int i, string prefix, string postfix);



int main() {	
		
	/*double a[9] = {3, 2, 5, 8, 4, 1, 2, 2, 6};
	double b[3] = {8, 1, 7};
	double c[3] = {23, 32, 2};
	multiplyM33M31(a, b, c);
	
	for (int i=0; i<3; i++) {
		cout << c[i] << " ";
	}
	cout <<endl;
	system("pause");
	
	exit(0);
	*/
	srand((unsigned)time(NULL));		
	
	Network yarp; 
	
	dummyInitialization();
	
	MySim *mysim = new MySim();	
		
	mysim->deleteAllObjects();	
	mysim->putObjectBeforeEye_left();				
		
	cout << " ready... " << endl;
	system("pause");

	string datasetDir = "../../data/130318/";
			
	
	ofstream objects_file, eyes_file;	
			
	objects_file.open((datasetDir + "_objects.txt").c_str());
	eyes_file.open((datasetDir + "_eyes.txt").c_str());

	mysim->setOutput(&objects_file);

	for (int i=0; i<1500; ) {		
		
		mysim->deleteAllObjects();                    // Mazanie objektu
		
		double tilt = MU::fRand(-35.0, 15.0);         
		double version = MU::fRand(-50.0, 50.0);
		
		mysim->turnEyes(tilt, version);               // Nastavenie pozicie oci na nahodnu poziciu v ramci intervalu
		mysim->putRandomObjectWRotation(true, 2);     // Nastavenie pozicie objecktu
				
		Sleep(300);
				
		ImageOf<PixelRgb> *imgLeft, *imgRight;        
		imgLeft = mysim->getLeftEyeImage();           // Obraz na lavej sosovke 
		imgRight = mysim->getRightEyeImage();         // Obraz na pravej sosovke
        
        /*
         * Uloz obrazok do datasetu
         */
		if (checkNotBlank(*imgLeft,0.05) && checkNotBlank(*imgRight,0.05)) {
			eyes_file << tilt << " " << version << endl;
			
			processImage(mysim->getLeftEyeImage(), i, datasetDir, "l");
			processImage(mysim->getRightEyeImage(), i, datasetDir, "r");
			
			i++;
		}

		//system("pause");
	}
	
	objects_file.close();
	mysim->disconnectLeftEye();
	mysim->disconnectRightEye();

	cout << "done...";
	system("pause");

	delete mysim;		
	
	return 0;
 }

 
/*
 * Spracovanie snimkov
 */
void processImage(ImageOf<PixelRgb> *img, int index, string prefix, string postfix) {		
								
		//procesed image
		int nscale = 5;									
		int iw = 320/nscale;
		int ih = 240/nscale;
		int *mat = new int[iw*ih];
		for (int i = 0; i<iw*ih; i++) { 
			mat[i] = 0; 
		}

		int maxv = 1;
		for (int x=0; x<img->width(); x++) {
			for (int y=0; y<img->height(); y++) {
				PixelRgb &pix = img->pixel(x,y);				
				if (pix.r>pix.b*1.2+10 && pix.r>pix.g*1.2+10) {
					mat[(y/nscale)*iw + x/nscale] += pix.r;					
					maxv = (pix.r > maxv) ? pix.r : maxv;
				} 
			}
		}	

		ImageOf<PixelRgb> *obr = new ImageOf<PixelRgb>();				
		obr->resize(iw, ih);
		double d = 255.0 / (25.0*(double)maxv);
		for (int x=0; x<obr->width(); x++) {
			for (int y=0; y<obr->height(); y++){
				PixelRgb &p = obr->pixel(x, y);
				p.r = (int)(mat[y*iw + x] * d);								
				p.g = p.r;
				p.b = p.r;				
			}
		}		

		
		//save original image
		string filename = prefix;				
		filename += "o_" + to_string((long double)index) + postfix + ".ppm";
		yarp::sig::file::write(*img, filename.c_str());
		
		filename = prefix;
		filename += to_string((long double)index) + postfix + ".ppm";
		yarp::sig::file::write(*obr, filename.c_str());		
		
 }
