
#include "doublefann.h"
#include "fann_cpp.h"

#include "MyUtils.h"
#include "NeuronCoders.h"
#include "MyCommon.h"

#include <yarp/sig/Image.h>
#include <yarp/sig/ImageFile.h>


#include <ios>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
using namespace std;
using namespace yarp::sig;

ofstream ofs;


// Callback function that simply prints the information to cout
int print_callback(FANN::neural_net &net, FANN::training_data &train,
    unsigned int max_epochs, unsigned int epochs_between_reports,
    float desired_error, unsigned int epochs, void *user_data)
{  	
	cout << "Epochs     " << setw(8) << epochs << ". "
         << "Current Error: " << left << net.get_MSE() << right << endl;
	
	if (ofs.is_open()) {
		ofs << "Epochs     " << setw(8) << epochs << ". "
			 << "Current Error: " << left << net.get_MSE() << right << endl;
	}
    
	return 0;
}


/*
Nacita siet a vypise vahy spojeni zo skrytej vrstvy na vstupnu a z vystupnej na skrytu.
*/
void processNet(string networkFile, string configFile, string outputDir, bool with_weight_factors) {
	cout << "Reading config file " << configFile << endl;
		
	//load tilt and version
	Parameters params = loadParameters(configFile);
		
	cout << "Creating neural network: " << networkFile << endl;
	FANN::neural_net net;	 		
	//WEIGHT FACTOR	
	net.create_from_file_with_weight_factors(networkFile, with_weight_factors);	
	//net.create_from_file(networkFile);

	unsigned int totalCon = net.get_total_connections();
	cout << "total connections " << totalCon << endl;
	FANN::connection *connections = new FANN::connection[totalCon];
	net.get_connection_array(connections);
	 	 
	unsigned int layersCount = net.get_num_layers();
	unsigned int *layersNums = new unsigned int[layersCount];
	net.get_layer_array(layersNums);	 
	cout << "layers nums: " << layersNums[0] << " " << layersNums[1] << " " << layersNums[2] << endl;
				
	
	unsigned int retinaNum = params.retinaWidth*params.retinaHeight;
		


	cout << "Processing hidden neuron: " <<endl;
	
	ofstream ofsBias;
	ofsBias.open( (outputDir + "hbiases").c_str());

	//spracujem kazdy neuron na skrytej vrstve
	for (unsigned int i=0; i<layersNums[1]; i++) {


		cout << i << endl;
	
		#ifdef _MSC_VER
			#pragma warning(disable: 4996)
		#endif
	

		ofstream ofsl;
		char outputFile[100]; //= "../../data/visualisation/hid301/lhid." + to_string((long double)i);
		sprintf(outputFile, (outputDir + "lhid.%03d").c_str(), i);				
		ofsl.open(outputFile);			

		ofstream ofsr;
		//string outputFileR = "../../data/visualisation/hid301/rhid." + to_string((long double)i);
		sprintf(outputFile, (outputDir + "rhid.%03d").c_str(), i);
		ofsr.open(outputFile);	

		ofstream ofse;		
		//string outputFileE = "../../data/visualisation/hid301/thid." + to_string((long double)i);
		sprintf(outputFile, (outputDir + "thid.%03d").c_str(), i);
		ofse.open(outputFile);	

		#ifdef _MSC_VER
			#pragma warning(default: 4996)
		#endif

		int x=0, y=1;	
		
		unsigned int base = (layersNums[0]+1)*i;	
		int e = 0;
		double w = 0;

		for (unsigned int j=base; j<base + params.tiltNum + params.versionNum + retinaNum; j++) {
			
			if (j<base+params.tiltNum) {
				e++;
				w = connections[j].weight;
				//w = connections[j].weight > 1000 ? 0 : connections[j].weight;
				MU::printlnValues(ofse, "idii", e, w, connections[j].from_neuron, connections[j].to_neuron);				
			} else if (j<base + params.tiltNum + params.versionNum) {
				if (j==base + params.tiltNum) {
					e=0;
					ofse.close();
					//outputFileE = "../../data/visualisation/hid301/vhid." + to_string((long double)i);
					sprintf(outputFile, (outputDir + "vhid.%03d").c_str(), i);
					ofse.open(outputFile);
				}
				e++;
				w = connections[j].weight;
				MU::printlnValues(ofse, "idii", e, w, connections[j].from_neuron, connections[j].to_neuron);								
			} else {
				x++;						
				
				w = min(max(connections[j].weight, -500), 500);
				ofsl << x << " " << y << " " << w << " " << connections[j].from_neuron << " " << connections[j].to_neuron << endl;			
				int k = j + retinaNum;
				w = min(max(connections[k].weight, -500), 500);
				ofsr << x << " " << y << " " << w << " " << connections[k].from_neuron << " " << connections[k].to_neuron << endl;			

				if (x==params.retinaWidth) {
					x=0;
					y++;
				}			
			} 
			
		}
		ofse.close();
		ofsl.close();
		ofsr.close();

		e  = base + params.tiltNum + params.versionNum + retinaNum*2;
		MU::printlnValues(ofsBias, "dii", connections[e].weight, connections[e].from_neuron, connections[e].to_neuron);		
			
	}	 
	ofsBias.close();
	

	//spracujem kazdy neuron na skrytej vrstve
	unsigned int base = layersNums[1]*(layersNums[0]+1);

	cout << "output layer: " << endl;

	for (unsigned int i=0; i<layersNums[2]; i++) {
		cout << i << endl;
		ofstream ofs;
		char outputFile[100]; 
		sprintf(outputFile, (outputDir + "out.%03d").c_str(), i);
		ofs.open(outputFile);	

		for (unsigned int j=0; j<layersNums[1]+1; j++) {
			unsigned int index = base+i*(layersNums[1]+1)+j;		
			MU::printlnValues(ofs, "dii", connections[index].weight, connections[index].from_neuron, connections[index].to_neuron);
		}

		ofs.close();

	}

	delete[] layersNums;
	 
}


void processNet2(string networkFile, string configFile, string outputDir, bool with_weight_factors) {
	cout << "Reading config file " << configFile << endl;
		
	//load tilt and version
	Parameters params = loadParameters(configFile);
		
	cout << "Creating neural network: " << networkFile << endl;
	FANN::neural_net net;	 	
	//WEIGHT_FACTOS	
	net.create_from_file_with_weight_factors(networkFile, with_weight_factors);	
	//net.create_from_file(networkFile);

	unsigned int totalCon = net.get_total_connections();
	cout << "total connections " << totalCon << endl;
	FANN::connection *connections = new FANN::connection[totalCon];
	net.get_connection_array(connections);
	 	 
	unsigned int layersCount = net.get_num_layers();
	unsigned int *layersNums = new unsigned int[layersCount];
	net.get_layer_array(layersNums);	 
	cout << "layers nums: " << layersNums[0] << " " << layersNums[1] << " " << layersNums[2] << endl;
				
	
	unsigned int retinaNum = params.retinaWidth*params.retinaHeight;
		


	cout << "Processing hidden neuron: " <<endl;
	
	ofstream ofsBias;
	ofsBias.open( (outputDir + "hbiases").c_str());

	//spracujem kazdy neuron na skrytej vrstve
	for (unsigned int i=0; i<layersNums[1]; i++) {


		cout << i << endl;
	
		#ifdef _MSC_VER
			#pragma warning(disable: 4996)
		#endif
	

		ofstream ofsl;
		char outputFile[100]; //= "../../data/visualisation/hid301/lhid." + to_string((long double)i);
		sprintf(outputFile, (outputDir + "lhid.%03d").c_str(), i);				
		ofsl.open(outputFile);			

		ofstream ofsr;
		//string outputFileR = "../../data/visualisation/hid301/rhid." + to_string((long double)i);
		sprintf(outputFile, (outputDir + "rhid.%03d").c_str(), i);
		ofsr.open(outputFile);	

		ofstream ofse;		
		//string outputFileE = "../../data/visualisation/hid301/thid." + to_string((long double)i);
		sprintf(outputFile, (outputDir + "thid.%03d").c_str(), i);
		ofse.open(outputFile);	

		#ifdef _MSC_VER
			#pragma warning(default: 4996)
		#endif

		int x=0, y=1;	
		
		unsigned int base = (layersNums[0]+1)*i;	
		int e = 0;
		double w = 0;

		for (unsigned int j=base; j<base + params.tiltNum + params.versionNum + retinaNum; j++) {
			
			if (j<base+params.tiltNum) {
				e++;
				w = connections[j].weight;
				//w = connections[j].weight > 1000 ? 0 : connections[j].weight;
				MU::printlnValues(ofse, "idii", e, w, connections[j].from_neuron, connections[j].to_neuron);				
			} else if (j<base + params.tiltNum + params.versionNum) {
				if (j==base + params.tiltNum) {
					e=0;
					ofse.close();
					//outputFileE = "../../data/visualisation/hid301/vhid." + to_string((long double)i);
					sprintf(outputFile, (outputDir + "vhid.%03d").c_str(), i);
					ofse.open(outputFile);
				}
				e++;
				w = connections[j].weight;
				MU::printlnValues(ofse, "idii", e, w, connections[j].from_neuron, connections[j].to_neuron);								
			} else {
				x++;						
				
				w = min(max(connections[j].weight, -500), 500);
				ofsl << x << " " << y << " " << w << " " << connections[j].from_neuron << " " << connections[j].to_neuron << endl;			
				int k = j + retinaNum;
				w = min(max(connections[k].weight, -500), 500);
				ofsr << x << " " << y << " " << w << " " << connections[k].from_neuron << " " << connections[k].to_neuron << endl;			

				if (x==params.retinaWidth) {
					x=0;
					y++;
				}			
			} 
			
		}
		ofse.close();
		ofsl.close();
		ofsr.close();

		e  = base + params.tiltNum + params.versionNum + retinaNum*2;
		MU::printlnValues(ofsBias, "dii", connections[e].weight, connections[e].from_neuron, connections[e].to_neuron);		
			
	}	 
	ofsBias.close();
	

	//spracujem kazdy neuron na skrytej vrstve
	unsigned int base = layersNums[1]*(layersNums[0]+1);

	cout << "output layer: " << endl;

	for (unsigned int i=0; i<layersNums[2]; i++) {
		cout << i << endl;
		ofstream ofs;
		char outputFile[100]; 
		sprintf(outputFile, (outputDir + "out.%03d").c_str(), i);
		ofs.open(outputFile);	

		for (unsigned int j=0; j<layersNums[1]+1; j++) {
			unsigned int index = base+i*(layersNums[1]+1)+j;		
			MU::printlnValues(ofs, "dii", connections[index].weight, connections[index].from_neuron, connections[index].to_neuron);
		}

		ofs.close();

	}

	delete[] layersNums;
	 
}


void visualiseDataSet(string fileDataset, string fileParameters, string outputDir) {
	cout << "Creating images from dataset " << fileDataset << ". Target directory: " << outputDir << endl;

	Parameters params = loadParameters(fileParameters);

	FANN::training_data testData;	 
	testData.read_train_from_file(fileDataset.c_str());
	
	char *idx_string = new char[3];
	for (unsigned int idx=0; idx<testData.length_train_data(); idx++) {				
		fann_type *input = testData.get_input()[idx];

		ImageOf<PixelRgb> *img = new ImageOf<PixelRgb>();
		img->resize(params.retinaWidth, params.retinaHeight);						
		unsigned index = params.tiltNum+params.versionNum;
		for (unsigned int y=0; y<params.retinaHeight; y++) {
			for (unsigned int x=0; x<params.retinaWidth; x++) {				
				PixelRgb &pix = img->pixel(x,y);
				int val = input[index]*255;
				pix.r = val;
				pix.g = val;
				pix.b = val;		
				index++;						
			}				
		}		

		sprintf(idx_string, "%03d", idx);
		string file = outputDir + idx_string + "img_l.ppm";
		cout << file << endl;
		yarp::sig::file::write(*img, file.c_str());		
		delete img;		
	}
}


void saveTilts(Parameters params, const int tn, double tilts[], string outputFile) {					
	ofstream ofs;
	ofs.open(outputFile);
	double* tilt = new double[params.tiltNum];
	for (int t=0; t<tn; t++) {						
			NeuronCoders::valueToNeuronArray(tilts[t], params.tiltNum, params.tiltPeaks, tilt, params.tiltGaussianHeight, params.tiltGaussianWidth);
			ofs << tilts[t] << " ";
			MU::printlnArr(ofs, tilt, params.tiltNum);
	}
	delete tilt;
	ofs.close();
}

void saveVersions(Parameters params, const int vn, double versions[], string outputFile) {			
	ofstream ofs;
	ofs.open(outputFile);
	double* version = new double[params.versionNum];
	for (int v=0; v<vn; v++) {										
			NeuronCoders::valueToNeuronArray(versions[v], params.versionNum, params.versionPeaks, version, params.versionGaussianHeight, params.versionGaussianWidth);																								
			ofs << versions[v] << " ";
			MU::printlnArr(ofs, version, params.versionNum);
	}
	delete[] version;
	ofs.close();
}

void saveChoosen(FANN::training_data testData, Parameters params, unsigned int choosenCount, unsigned int choosen[], string outputFile) {
		
	ofstream ofs, ofsall;

	ofsall.open((outputFile+"img_all").c_str());

	// pre vsetky obrazky
	for (unsigned int idx = 0; idx<choosenCount; idx++) {	
		unsigned int p = choosen[idx];		
	
		char *p_string = new char[3];
		sprintf(p_string, "img%03d", p);
		ofs.open((outputFile + p_string).c_str());

		// nacitaj original input
		fann_type *iii = testData.get_input()[p];
		fann_type *start = iii+params.versionNum+params.tiltNum;
		for (unsigned int row=0; row<params.retinaHeight; start += params.retinaWidth, row++) {
			MU::printlnArr(ofs, start, params.retinaWidth);
			MU::printlnArr(ofsall, start, params.retinaWidth);
		}
		ofs << endl;
		ofsall << endl;
		for (unsigned int row=0; row<params.retinaHeight; start += params.retinaWidth, row++) {
			MU::printlnArr(ofs, start, params.retinaWidth);
			MU::printlnArr(ofsall, start, params.retinaWidth);
		}
		ofsall << endl;
		ofsall << endl;						

		ofs.close();
	}
	
	ofsall.close();

	ofs.open((outputFile+"choosen").c_str());	
	string a = "";
	for (int i=0; i<choosenCount; i++) {
		ofs << a << choosen[i];
		a = " ";
	}	
	ofs.close();
}

/*
	Pre konkretny obrazok nastavi tilt na 0 a zaznamena vystupy zo skrytej vrstvy pre version: -40, -20, 0, 20, 40
*/
void testGainFields(string fileNet, string fileDataset, string fileConfig, string outputDir, unsigned int choosenCount, unsigned int choosen[], bool with_weight_factors) {
	 	 
	
	//nacitanie parametrov siete - aby som vedel generovat vstupy

	cout << "Loading configuration: " << fileConfig << endl; 
	
	Parameters params = loadParameters(fileConfig);

	// nacitanie dat - zoberu sa realne obrazky
	cout << "Loading test data: " << fileDataset << endl;
	FANN::training_data testData;	 
	testData.read_train_from_file(fileDataset.c_str());

	cout << "Saving choosen..." << endl;
	saveChoosen(testData, params, choosenCount, choosen, outputDir);
	
	//unsigned int choosen = 11;
	
	unsigned int inputNum = testData.num_input_train_data();


	fann_type *input = new fann_type[inputNum+1];
	
		
	// nacitanie natrenovanej siete
	cout << "Creating neural network: " << fileNet << endl;
	FANN::neural_net net;	 	
    //WEIGHT FACTOR	
	net.create_from_file_with_weight_factors(fileNet, with_weight_factors);	
	//net.create_from_file(fileNet);
	cout << "Running..." << endl;

	// pre kazdy obrazkovy vstup vytvor tilt a version, uloz aktivacie skrytych a vystupnych neuronov do suboru
	const int tn = 5, vn = 5;
	double tilts[tn] = {-30,-20, -10, 0, 10};
	double versions[vn] = {-40, -20, 0, 20, 40};
	double *version = new double[params.versionNum];
	double *tilt = new double[params.tiltNum];
	
	saveTilts(params, tn, tilts, (outputDir+"tilts"));
	saveVersions(params, vn, versions, (outputDir+"versions"));	

	unsigned int *layers = new unsigned int[net.get_num_layers()];
	net.get_layer_array(layers);	
	unsigned int hiddenNum = layers[1];
	double *activations = new double[hiddenNum];
	double *sums = new double[hiddenNum];

		
	ofstream ofsActivations, ofsSums, ofsAllActivations, ofsAllSums;
	
	ofsAllActivations.open((outputDir+"pall_activations").c_str());
	ofsAllSums.open((outputDir+"pall_sums").c_str());

	unsigned int outputNum = net.get_num_output();
	
	char *del = " ";


	// pre vsetky obrazky
	for (unsigned int idx = 0; idx<choosenCount; idx++) {	
		unsigned int p = choosen[idx];		

		char *p_string = new char[3];
		sprintf(p_string, "%03d", p);

		ofsActivations.open((outputDir + "p" + p_string + "_activations").c_str());
		ofsSums.open((outputDir+"p" + p_string + "_sums").c_str());
				
		// nacitaj original input
		fann_type *iii = testData.get_input()[p];
		for (unsigned int i=0; i<inputNum+1; i++) {		
			input[i] = iii[i];
		}		

		// pre vsetky vertikalne pozicie
		for (int t=0; t<tn; t++) {
			
			// preved dany uhol do population codu a nastav ho ako input
			NeuronCoders::valueToNeuronArray(tilts[t], params.tiltNum, params.tiltPeaks, tilt, params.tiltGaussianHeight, params.tiltGaussianWidth);			
			for (unsigned int i=0; i<params.tiltNum; i++) {				
				input[i] = tilt[i];
			}			

			// pre vsetky horizontalne pozicie
			for (int v=0; v<vn; v++) {			
				
				// preved dany uhol do population codu a nastav ho ako input
				NeuronCoders::valueToNeuronArray(versions[v], params.versionNum, params.versionPeaks, version, params.versionGaussianHeight, params.versionGaussianWidth);																		
				for (unsigned int i=params.tiltNum;  i<params.tiltNum+params.versionNum; i++) {
					input[i] = version[i-params.tiltNum];
				}
				
				// spust siet na danom vstupe
				fann_type *real_output = net.run(input);

				// iba kvoli overeniu si zistim co siet vypocitala
				double xreal = NeuronCoders::neuronArrayToValue(real_output, params.xNum, params.xPeaks, params.xGaussianHeight, params.xGaussianWidth);
				double yreal = NeuronCoders::neuronArrayToValue((real_output+params.xNum), params.yNum, params.yPeaks, params.yGaussianHeight, params.yGaussianWidth);

				net.get_last_activations(1, activations);
				net.get_last_sums(1, sums);

				ofsActivations << tilts[t] << del << versions[v] << del;									
				ofsSums << tilts[t] << del << versions[v] << del;

				MU::printArr(ofsActivations, activations, hiddenNum, del);
				ofsActivations << del;
				MU::printArr(ofsActivations, real_output, outputNum, del);

				MU::printlnArr(ofsAllActivations, activations, hiddenNum, del);
								
				MU::printArr(ofsSums, sums, hiddenNum, del);
				ofsSums << del;			
				MU::printArr(ofsSums, real_output, outputNum, del);

				MU::printlnArr(ofsAllSums, sums, hiddenNum, del);

				
				ofsActivations << del << xreal << del << yreal << del << p << endl;
				ofsSums << del << xreal << del << yreal << del << p << endl;
			}
		}
		
		for (unsigned int i=0; i<hiddenNum+net.get_num_output()+3; i++) {
			ofsActivations << del ;
			ofsSums << del ;
		}				
		ofsActivations << endl;
		ofsSums << endl;

		ofsActivations.close();
		ofsSums.close();
	
	}

	ofsAllActivations.close();
	ofsAllSums.close();
	
	delete[] input;
	delete[] version;	
	delete[] tilt;



//	ofstream ofs;
	//ofs.open(outputFile.c_str());			
	

	/*
		ofstream ofs;
	ofs.open(outputFile.c_str());
	ofs << net.get_last_activations(1, NULL);
	ofs.close();


	/*
	net.reset_MSE();
	
	
	unsigned int choosen = 30;

	

	//TODO - meranie vystupov na skrytej vrstve vo fann
	*/
}


/* Startup function. Syncronizes C and C++ output, calls the test function
   and reports any exceptions */
int main(int argc, char **argv) {    
	
	//string root = "../../data/";
	
	string root="", net = "", parameters ="", outputDir = "", data ="";
	
	
	/*
	root = "../../data/130203-balls/";
	net = root + "3_nets/net100inc.net";
	string data = root + "2_processed_fann/data_test.txt";
	parameters = root + "2_processed_fann/data_parameters.txt";
	
	//outputDir = root + "4_visualisations/net100inc/";			
	//processNet(root+net, root+parameters, root+outputDir, false);

	outputDir = root + "5_gain_fields/net100inc";
	testGainFields(net, data, parameters, outputDir, 74, false);				
	*/

	//______ net100inc-fb.net ______
	
	/*root = "../../data/130203-balls/";
	string netname = "net100rprop-fb2";
	net = root + "3_nets/"+netname+".net";
	outputDir = root + "6_all_in/"+netname+"/data/";			
	parameters = root + "2_processed_fann/data_parameters.txt";
	processNet(net, parameters, outputDir, true);
	data = root + "2_processed_fann/data_test.txt";
	unsigned int patterns[9] = {139,128,136,412,454,144,27,387,316};
	testGainFields(net, data, parameters, outputDir, 9, patterns, true);				
	*/

	/*
	root = "../../data/130203-balls/";
	string netname = "net100inc";
	net = root + "3_nets/"+netname+".net";
	outputDir = root + "6_all_in/"+netname+"/data/";			
	parameters = root + "2_processed_fann/data_parameters.txt";
	processNet(net, parameters, outputDir, false);
	data = root + "2_processed_fann/data_test.txt";
	unsigned int patterns[9] = {139,128,136,412,454,144,27,387,316};
	testGainFields(net, data, parameters, outputDir, 9, patterns, false);				
	*/
	
	//______ 19_net100inc-fb.net ______	

	//saveTilts("../../data/130318/2_processed/19_parameters.txt","../../data/130318/2_processed/population_codes/19_tilts");
	//saveVersions("../../data/130318/2_processed/19_parameters.txt","../../data/130318/2_processed/population_codes/19_versions");
	//unsigned int patterns[9] = {238, 260, 119, 14, 139, 48, 214, 59, 35};
	//saveChoosen("../../data/130318/2_processed/19_test.txt", "../../data/130318/2_processed/19_parameters.txt", 9, patterns, "../../data/130318/2_processed/choosen_inputs/");
	//exit(0);
	
	root = "../../data/130318-R/";	
	string datasetname = "19";
	string netname = datasetname + "_net100incm10-fb";	
	net = root + "3_nets/"+netname+".net";
	outputDir = root + "6_all_in/"+netname+"/";			
	parameters = root + "2_processed/"+datasetname+"_parameters.txt";
	processNet(net, parameters, outputDir, true);		
	data = root + "2_processed/"+datasetname+"_test.txt";	
	unsigned int patterns[9] = {238, 260, 119, 14, 139, 48, 214, 59, 35};
	testGainFields(net, data, parameters, outputDir, 9, patterns, true);	
	

	//______ net100inc-fb2.net Receptive field ______
/*
	root = "../../data/130203-balls/";
	net = "3_nets/net100inc-fb2.net";
	outputDir = "4_visualisations/net100inc-fb2/";			
	parameters = "2_processed_fann/data_parameters.txt";
	processNet(root+net, root+parameters, root+outputDir, true);


	//______ net100inc-fb2.net Gain field ______
	
	root = "../../data/130203-balls/";
	data = root + "2_processed_fann/data_test.txt";
	net = root + "3_nets/net100inc-fb2.net";
	parameters = root + "2_processed_fann/data_parameters.txt";
	outputDir = root + "5_gain_fields/net100inc-fb2/";
	unsigned int patterns[9] = {139,128,136,412,454,144,27,387,316};
	testGainFields(net, data, parameters, outputDir, 9, patterns, true);				
*/
	//______ net100rprop-fb2.net Receptive field ______
	/*	
	root = "../../data/130203-balls/";
	net = "3_nets/net100rprop-fb2.net";
	outputDir = "4_visualisations/net100rprop-fb2/";			
	parameters = "2_processed_fann/data_parameters.txt";
	processNet(root+net, root+parameters, root+outputDir, true);
	*/

	//outputDir = root + "4_visualisations/net100inc-fb/";			
	//processNet(root+net, root+parameters, root+outputDir, true);

	//outputDir = root + "5_gain_fields/net100inc-fb";
	


	/*
	root = "../../data/130203-balls/";
	net = root + "3_nets/net100inc-fb.net";
	string data = root + "2_processed_fann/data_test.txt";
	parameters = root + "2_processed_fann/data_parameters.txt";
	
	//outputDir = root + "4_visualisations/net100inc-fb/";			
	//processNet(root+net, root+parameters, root+outputDir, true);

	outputDir = root + "5_gain_fields/net100inc-fb";
	testGainFields(net, data, parameters, outputDir, 74, true);	
	*/
/*
	cout << "----Net visualiser----"<<endl;
	cout << endl << "Root dir: ";
	cin >> root;
	cout << endl << "Net file: ";
	cin >> net;
	cout << endl << "Parameters file (to load tilt and version nums): ";
	cin >> parameters;
	cout << endl << "Output directory (ends with /): ";
	cin >> outputDir;
*/
/*	root = "../../data/130203-balls/";
	net = "3_nets/net100inc-fb.net";
	parameters = "2_processed_fann/data_parameters.txt";
	outputDir = "4_visualisations/net100inc-fb/";
	
	processNet(root+net, root+parameters, root+outputDir, true);
	*/	
	system("pause");
	//MU::pause();
    return 0;
}
