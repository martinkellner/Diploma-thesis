
#include "doublefann.h"
#include "fann_cpp.h"

#include "MyUtils.h"
#include "NeuronCoders.h"
#include "MyCommon.h"

#include <ios>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
using namespace std;

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


string getCurrentTime() {
	// Current date/time based on current system
	time_t now = time(0);

	// Convert now to tm struct for local timezone
	tm* localtm = localtime(&now);
	cout << "The local date and time is: " << asctime(localtm) << endl;

	// Convert now to tm struct for UTC
	tm* gmtm = gmtime(&now);
	cout << "The UTC date and time is: " << asctime(gmtm) << endl;

	return "";
}




void calculateFactors(double realA, double realB, double desA, double desB, double &factorA, double &factorB) {
	double 		
		realAll = (realA + realB), 
		desAll = (desA + desB);
	factorA = (desA * realAll) / (desAll * realA);
	factorB = (desB * realAll) / (desAll * realB);
}


/*
 10.3. - rozdelujem input na 2 sekcie - nakoniec moze mat kazdy neuron inu vahu na kazdy vstupny neuron
*/
void dummyTest() {
	FANN::neural_net net;

	unsigned int input = 10, hidden = 5, output =3;

	unsigned int layers[3] = {input, hidden, output};
	net.create_standard_array(3, layers);

	net.set_learning_rate(0.2);
	
	net.set_training_algorithm(FANN::TRAIN_INCREMENTAL);
	net.set_train_error_function(FANN::ERRORFUNC_LINEAR);
	net.set_activation_function_hidden(FANN::SIGMOID);
    net.set_activation_function_output(FANN::SIGMOID);
	double activation_steepness_h = 1;
	net.set_activation_steepness_hidden(activation_steepness_h);
	//unsigned int num_connections = hidden*(input+1);
	//FANN::connection *connections = new FANN::connection[num_connections];		
	
	//cout << "Total connections: " << net.get_total_connections() << " | test spravnosti: " << num_connections + (hidden+1)*output << endl;

	

	unsigned int total_con = net.get_total_connections();
	FANN::connection *con = new FANN::connection[total_con];	
	net.get_connection_array(con);
	
	//WEIGHT_FACTORS
	for (unsigned int idx = 0; idx<total_con; idx++) {
		con[idx].weight = 0.5;
		
		if ( (con[idx].from_neuron<5) &&  (con[idx].to_neuron == (input+1)+hidden-1)) {
			con[idx].weight_factor = 3.232;
		}
			
	}
	net.set_weight_factor_array(con, total_con);	
	net.set_weight_array(con, total_con);
	
	
	string netFile = "../../data/dummy/netww";
		 
	//WEIGHT_FACTORS 
	net.save_with_weight_factors(netFile);

	fann_type *in = new fann_type[input];
	for (unsigned int idx=0; idx<input; idx++) {
		in[idx] = idx/10.0;
	}
	fann_type *out;
	out = net.run(in);	

	
	
	cout << "steepness " << net.get_activation_steepness(1,1) << endl;
	cout << "hidden" << endl;
		
	double *activations = new double[hidden];
	net.get_last_activations(1, activations);
	

	double *sums = new double[hidden];
	net.get_last_sums(1, sums);
	
	MU::printlnArr(cout, sums, hidden); 	
	MU::printlnArr(cout, activations, hidden); 		
	MU::printlnArr(cout, out, output);


	
	double th_sum = 0, th_activation = 0;
	for (unsigned int idx=0; idx<input; idx++) {		
		double d = in[idx]*0.5;
		if (idx<5) {
			d*=3;
		}
		cout << "in[idx] " <<  in[idx] << " d: "  << d << "  " << th_sum << endl;
		th_sum += d;
	}
	
	th_sum += 0.5;
	th_sum *= activation_steepness_h;
	th_activation = 1 / (1 + exp(-2.0*th_sum));
	
	cout << "calculated " << endl;
	MU::printlnValues(cout, "dd", th_sum, th_activation);


	FANN::neural_net net2;
//WEIGHT_FACTORS	
	net2.create_from_file_with_weight_factors(netFile, false);
	//net.create_from_file(netFile);
	out = net2.run(in);
	MU::printlnArr(cout, out, output);



	/*
	for (unsigned int h=0; h<hidden; h++) {
		for (unsigned int i=0; i<=input; i++) {
			unsigned int idx = i+h*(input+1);			
			connections[idx].from_neuron = i;	
			connections[idx].to_neuron = h+(input+1);
			connections[idx].weight = 0.5;						
		}
	}
	*/
			
}


//rate 0.4
//hidden 300
//des error 0.002
//max it
void trainFANN(string networkName, string trainFile, string paramsFile, string alg, unsigned int num_hidden, float learning_rate, float momentum, double desired_error, unsigned int max_iterations, bool with_weight_factors) {    

	int t0 = time(NULL);
	
	string outFile = networkName+".log";
	ofs.open(outFile.c_str());
	    
    unsigned int 
		num_layers = 3,
		num_input = 0,		
		num_output = 0;

	ifstream ifs;
	ifs.open(trainFile.c_str());
	ifs >> num_input; //skip total number of patterns
	ifs >> num_input;
	ifs >> num_output;
	ifs.close();
    
	Parameters params = loadParameters(paramsFile);

    const unsigned int iterations_between_reports = 5;

    cout << endl << "Creating network." << endl;
	ofs << endl << "Creating network." << endl;

    FANN::neural_net net;

	if (alg=="cascade") {
		unsigned int layers[2] = {num_input, num_output}; 	
		//net.create_shortcut(2, num_input, num_output);
		net.create_shortcut_array(2, layers);		
	} else {
		unsigned int layers[3] = {num_input, num_hidden, num_output}; 
		net.create_standard_array(3, layers);
	}
	

//	unsigned int layers[4] = {num_input, 250, 60, num_output}; 
  //  net.create_standard_array(4, layers);
        
    net.set_learning_rate(learning_rate);	

    //net.set_activation_steepness_hidden(1.0);
    //net.set_activation_steepness_output(1.0);
	
		
	//net.set_activation_steepness_hidden(1.0/15.0);
	//net.set_activation_steepness_output(1.0/10.0);	    
    
	net.set_activation_steepness_hidden(1.0/20.0);
	net.set_activation_steepness_output(1.0/10.0);	    
    

    net.set_activation_function_hidden(FANN::SIGMOID);
    net.set_activation_function_output(FANN::SIGMOID);

    // Set additional properties such as the training algorithm
	if (alg == "incremental") {
		net.set_training_algorithm(FANN::TRAIN_INCREMENTAL);
		net.set_train_error_function(FANN::ERRORFUNC_LINEAR);
		net.set_learning_momentum(momentum);
	} else if (alg=="rprop") {
		net.set_training_algorithm(FANN::TRAIN_RPROP);	
		net.set_train_error_function(FANN::ERRORFUNC_TANH);
		net.set_rprop_delta_min(0.000001);
	} else if (alg=="quickprop") {
		net.set_training_algorithm(FANN::TRAIN_QUICKPROP);	
		net.set_train_error_function(FANN::ERRORFUNC_TANH);
	} else if (alg=="batch") {
		net.set_training_algorithm(FANN::TRAIN_BATCH);
		net.set_train_error_function(FANN::ERRORFUNC_LINEAR);	
	} else if (alg=="cascade") {
		net.set_training_algorithm(FANN::TRAIN_RPROP);
		net.set_activation_function_hidden(FANN::SIGMOID);
		//net.set_activation_function_output(FANN::LINEAR);
		net.set_train_error_function(FANN::ERRORFUNC_TANH);				
	} else {
		cout << "error - unknown training algorithm!";
		MU::pause();
		exit(1);
	}

	//WEIGHT_FACTORS
	if (with_weight_factors) {
		
		unsigned int total_con = net.get_total_connections();
		FANN::connection *connections = new FANN::connection[total_con];
		net.get_connection_array(connections);
	
		unsigned int retinaNum = params.retinaWidth * params.retinaHeight * 2;
		unsigned int eyeposNum = params.tiltNum + params.versionNum;

		//zelany pomer medzi poctom neuronov kodujucich retinu a poctom kodujucim polohu
		double factorRetina = 1.0, factorEyePosition = 1.0;
		calculateFactors(retinaNum, eyeposNum, 2, 1, factorRetina, factorEyePosition);

		cout << "Factor retina " << factorRetina << endl << "Factor Eye Position: " << factorEyePosition << endl;
		ofs << "Factor retina " << factorRetina << endl << "Factor Eye Position: " << factorEyePosition << endl;
	
	
	
		for (unsigned int idx = 0; idx<total_con; idx++) {				
			// spojenia tilt a version 
			if (connections[idx].from_neuron < eyeposNum) {
				connections[idx].weight_factor = factorEyePosition;						
			} 
			// spojenia na tilt a version
			else if (connections[idx].from_neuron >= eyeposNum && connections[idx].from_neuron < retinaNum + eyeposNum) {
				connections[idx].weight_factor = factorRetina;		
			} else {
				connections[idx].weight_factor  = 1.0;
			}
		}				
		net.set_weight_factor_array(connections, total_con);
	}
	

    // Output network type and parameters
    cout << endl << "Network Type                         :  ";
	ofs << endl << "Network Type                         :  ";
    switch (net.get_network_type()) {
    case FANN::LAYER:
        cout << "LAYER" << endl;
		ofs << "LAYER" << endl;
        break;
    case FANN::SHORTCUT:
        cout << "SHORTCUT" << endl;
		ofs << "SHORTCUT" << endl;
        break;
    default:
        cout << "UNKNOWN" << endl;
		ofs << "UNKNOWN" << endl;
        break;
    }

	
	cout << "Activation steepness hidden             :" << setw(8) << net.get_activation_steepness(1,0) << endl;
	cout << "Activation steepness output             :" << setw(8) << net.get_activation_steepness(2,0) << endl;
	ofs << "Activation steepness hidden             :" << setw(8) << net.get_activation_steepness(1,0) << endl;
	ofs << "Activation steepness output             :" << setw(8) << net.get_activation_steepness(2,0) << endl;	

    net.print_parameters();
	ofs << net.print_parameters2();

    cout << endl << "Training network on " << trainFile << endl;
	ofs << endl << "Training network on" << trainFile  << endl;		

	FANN::training_data data;
	if (data.read_train_from_file(trainFile.c_str())) {
        // Initialize and train the network with the data
        //net.init_weights(data);
		net.randomize_weights(-0.5, 0.5);

        cout << "Max Epochs " << setw(8) << max_iterations << ". "
            << "Desired Error: " << left << desired_error << right << endl;	
		ofs << "Max Epochs " << setw(8) << max_iterations << ". "
            << "Desired Error: " << left << desired_error << right << endl;	

        net.set_callback(print_callback, NULL);
		
		if (alg=="cascade") {
			cout << "Cascade training";
			ofs << "Cascade Training" << endl;
			net.cascadetrain_on_data(data, num_hidden, 1, desired_error);
		} else {
			net.train_on_data(data, max_iterations, iterations_between_reports, desired_error);
		}
		
               
        cout << endl << "Saving network." << endl;
		ofs << endl << "Saving network." << endl;

        // Save the network    
		outFile = networkName+".net";	
		
		//WEIGHT FACTORS		
		if (with_weight_factors) {
			net.save_with_weight_factors(outFile);
		} else {
			net.save(outFile.c_str()); 
		}
		
    }

	int t = time(NULL) - t0;
	ofs << endl << "Training took: " << (t/3600) << " hours " << (t%3600)/60 << " minutes " << (t%60) << " seconds." << endl;
	
	time_t tim = time(0);   // 
	ofs << asctime(std::localtime(&tim));

	// Restore old cout.
	ofs.close();
	//cout.rdbuf( oldCoutStreamBuf );

}

//read test file
//for every pattern compute error
//translate desired and real output to angle and compare
void test(string networkFile, string dataFile, string configFile, string outputFile, string filePositions, string fileSizes, string fileEyes, bool with_weight_factors) {
	 	 
	ofstream ofs, ofsRaw, ofsAs; //ofsAS - ofs Activations and Sums
	ofs.open(outputFile.c_str());	
	ofsRaw.open((outputFile + "_raw.csv").c_str());
	ofsAs.open((outputFile+"_as.csv"));


	 cout << "Loading configuration: " << configFile << endl;
	 
	 Parameters params = loadParameters(configFile);	
	 	 	 
	 ifstream ifsPositions, ifsSizes, ifsEyes;
	 ifsPositions.open(filePositions);
	 ifsSizes.open(fileSizes);
	 ifsEyes.open(fileEyes);

	 cout << "Loading test data: " << dataFile << endl;
	 FANN::training_data testData;	 
	 testData.read_train_from_file(dataFile.c_str());
	 	 
	 cout << "Creating neural network: " << networkFile << endl;
	 FANN::neural_net net;	 
	 
	 //net.create_from_file(networkFile);
	 //WEIGHT_FACTOR 
	 net.create_from_file_with_weight_factors(networkFile, with_weight_factors);
	 
	 
	 net.reset_MSE();
	 net.test_data(testData);	 	 
	 	 
	 cout << "MSE for all patterns: " << net.get_MSE() << endl << "         xMSE         yMSE   overallMSE     desiredX   estimatedX        diffX     desiredY   estimatedY        diffY" << endl;
	 //ofs << "MSE for all patterns: " << net.get_MSE() << endl << "         xMSE         yMSE   overallMSE     desiredX   estimatedX        diffX     desiredY   estimatedY        diffY" << endl;
	 ofs << "xMSE,yMSE,overallMSE,desiredX,estimatedX,diffX,desiredY,estimatedY,diffY,positionX,positionY,size,tilt,version" << endl;
	 
	 ofsRaw << ",";
	 for (unsigned int i=1; i<=params.xNum+params.yNum; i++) {
		ofsRaw << "," << i;
	 }
	 ofsRaw << ",X,Y" << endl;

	 unsigned int num_layers = net.get_num_layers();
	 unsigned int *layers = new unsigned int[num_layers];
	 net.get_layer_array(layers);	 

	 double *activations = new double[layers[1]];
	 double *sums = new double[layers[1]];

	 for (unsigned int i=0; i<testData.length_train_data(); i++) {
			
			/*cout << "desired output:" << endl;
			fann_type *desired_output = testData.get_output()[i];
			for (unsigned int j=0; j<xNum+yNum; j++) {
				if (desired_output[j]>0.01) {
					std::printf(" %3i: %8.3f ", j, desired_output[j]);
					//cout << " " << j << ": " << desired_output[j];
				}				
			}
			cout << endl << "real output:" << endl;				
			fann_type *real_output = net.run(testData.get_input()[i]);
			for (unsigned int j=0; j<xNum+yNum; j++) {
				if (real_output[j]>0.01) {
					std::printf(" %3i: %8.3f ", j, real_output[j]);
					//cout << " " << j << ": " << real_output[j];
				}				
			}
			*/
					
			net.reset_MSE();
			fann_type *desired_output = testData.get_output()[i];
			fann_type *real_output = net.test(testData.get_input()[i], testData.get_output()[i]);

			net.get_last_activations(1, activations);
			net.get_last_sums(1, sums);
			ofsAs << i << ";";
			MU::printlnArr(ofsAs, sums, layers[1], ";");
			ofsAs << i << ";";
			MU::printlnArr(ofsAs, activations, layers[1], ";");

			double xmse = 0, ymse = 0;
			double *diff = new double[params.xNum+params.yNum];
			
			for (unsigned int j=0; j<params.xNum+params.yNum; j++) {
				double jdiff = (real_output[j] - desired_output[j]);				
				if (j<params.xNum) {
					xmse += jdiff*jdiff;
				} else {
					ymse += jdiff*jdiff;
				}			
				diff[j] = abs(jdiff);
			}
			xmse /= params.xNum;
			ymse /= params.yNum;			
						
			double xdes = NeuronCoders::neuronArrayToValue(desired_output, params.xNum, params.xPeaks, params.xGaussianHeight, params.xGaussianWidth);			
			double xreal = NeuronCoders::neuronArrayToValue(real_output, params.xNum, params.xPeaks, params.xGaussianHeight, params.xGaussianWidth);
			
			double ydes = NeuronCoders::neuronArrayToValue((desired_output+params.xNum), params.yNum, params.yPeaks, params.yGaussianHeight, params.yGaussianWidth);			
			double yreal = NeuronCoders::neuronArrayToValue((real_output+params.xNum), params.yNum, params.yPeaks, params.yGaussianHeight, params.yGaussianWidth);
									
			MU::printlnValues(cout, "ddddddddd", xmse, ymse, net.get_MSE(), xdes, xreal, abs(xdes-xreal), ydes, yreal, abs(ydes-yreal));		
			double posx, posy, size, tilt, version;
			string dump;
			ifsPositions >> posx >> posy;
			ifsSizes >> dump >> size;	
			ifsEyes >> tilt >> version;
			MU::printlnCSValues(ofs, "dddddddddddddd", xmse, ymse, net.get_MSE(), xdes, xreal, abs(xdes-xreal), ydes, yreal, abs(ydes-yreal), posx, posy, size, tilt, version);
			
			ofsRaw << i << ",desired,";
			MU::printArr(ofsRaw, desired_output, params.xNum+params.yNum, ",");			
			ofsRaw << "," << xdes << "," << ydes << endl;			

			ofsRaw << i << ",real,";
			MU::printArr(ofsRaw, real_output, params.xNum+params.yNum, ",");
			ofsRaw << "," << xreal << "," << yreal << endl;			

			ofsRaw << i << ",difference,";
			MU::printArr(ofsRaw, diff, params.xNum+params.yNum, ",");
			ofsRaw << "," << abs(xdes-xreal) << "," << abs(ydes-yreal) << endl;			

	 }
	 
	 delete[] layers;
	 //delete[] activations;
	 //delete[] sums;

	 delete[] params.xPeaks;
	 delete[] params.yPeaks;
	 
	 ifsPositions.close();
	 ifsSizes.close();

	 ofs.close();
	 ofsRaw.close();
	 ofsAs.close();

	 testData.destroy_train();
}


/* Startup function. Syncronizes C and C++ output, calls the test function
   and reports any exceptions */
int main(int argc, char **argv) {    
	
	/*
	dummyTest();
	system("pause");
	exit(0);
	*/
		
	string action = "", name = "", data = "", config="", alg = "rprop", output ="", root="../../data/", positions="", sizes="", eyes="";
	int hidden = 100, maxit = 150;
	double alfa = 0.4, momentum = 0.0, err = 0.0004;

	string datasetName = "", datasetGroup = "", netname = "";
	bool with_weight_factors = true;

	if (true) {
					
		
		/*root = "../data/";		
		*/

		action = "train";
		datasetName = "19";
		datasetGroup = "130318-R";
		netname = "_net100incm10-fb";
		data = root + datasetGroup + "/2_processed/" + datasetName +"_train.txt";		
		config = root + datasetGroup + "/2_processed/" + datasetName +"_parameters.txt";	
		name = root + datasetGroup + "/3_nets/" + datasetName + netname;		
		alg = "incremental"; 
		hidden = 64;
		maxit = 1000;
		alfa = 1.5;
		err = 0.0005;	
		momentum = 0.9;
		with_weight_factors = true;
		
	
		action = "test";
		name = root + datasetGroup + "/3_nets/"+ datasetName + netname + ".net";			
		data = root + datasetGroup + "/2_processed/" + datasetName + "_test.txt";				
		config = root + datasetGroup + "/2_processed/" + datasetName +"_parameters.txt";					
		output = root + datasetGroup + "/3_nets/" + datasetName + netname +".csv";
		positions = root + datasetGroup + "/2_processed/"+datasetName+"_test_positions.txt";
		sizes = root + datasetGroup + "/2_processed/"+datasetName+"_test_sizes.txt";		
		eyes = root + datasetGroup + "/2_processed/"+datasetName+"_test_eyes.txt";
		with_weight_factors = true;
	
		
		

	} else if (argc==1) {	
		cout << "Command line options:" << endl << endl;
		cout << " # Create and train 3-layers neural network:" << endl << endl;
		cout << "     train <network_name> <train_data_file> [<algorithm> [<hidden> [<alfa> [<desired_error> [<max_iterations>]]]]]" << endl << endl;
		cout << " # Test existing neural network:" << endl << endl;
		cout << "     test <network_file> <test_data_file> <config_file> " << endl << endl;
		cout << " # algorithm:      incremental, rprop, quickprop               |  default: " << alg << endl;
		cout << " # hidden:         number of neurons in the hidden layer       |  default: " << hidden << endl;
		cout << " # alfa:           learning rate                               |  default: " << alfa << endl;
		cout << " # desired_error:  desired MSE on all training patterns        |  default: " << err << endl;
		cout << " # max_iterations: limit maximum number of epochs              |  default: " << maxit << endl;
		
		cout << endl << "action ('train' or 'test'): ";	
		cin >> action;
		cout << "network file: "; 
		cin >> name;
		if (action == "train") {
			cout << "train data: ";
			cin >> data;
			cout << "algorithm: ";
			cin >> alg;
			cout << "hidden: ";
			cin >> hidden;
			cout << "alfa: ";
			cin >> alfa;
			cout <<  "derired error: ";
			cin >> err;
			cout << "max iterations: ";
			cin >> maxit;
		} else if (action == "test") {
			cout << "Root directory (optional): ";
			cin >> root;
			cout << "test data: ";
			cin >> data;
			cout << "parameters file: ";
			cin >> config;
			cout << "output file: ";
			cin >> output;
		} else {
			cout << endl << "error: Unknown action!" << endl;			
			return 1;
		}		

	} else if (argc>=6) {
		cout << argc << endl;
		action = argv[1];
		name = argv[2];
		data = argv[3];
		if (action=="test") {
			config = argv[4];
			output = argv[5];
		} else {
			if (argc>4) alg = argv[4];
			if (argc>5) hidden = atoi(argv[5]);
			if (argc>6) alfa = atof(argv[6]);
			if (argc>7) err = atof(argv[7]);
			if (argc>8) maxit = atoi(argv[8]);		
		}
	} else {
		cout << "Invalid number of parameters!"  << endl;		
		return 1;
	}
				
	try {
        std::ios::sync_with_stdio(); // Syncronize cout and printf output        
		
		if (action=="train") {
			trainFANN(name, data, config, alg, hidden, alfa, momentum, err, maxit, with_weight_factors);
			//MU::printlnValues(cout, "ssssiddi", "training", name.c_str(), data.c_str(), alg.c_str(), hidden, alfa, err, maxit);			
		} else if (action=="test") {
			test(name, data, config, output, positions, sizes, eyes, with_weight_factors);
			//MU::printlnValues(cout, "sss", "testing", name.c_str(), data.c_str());			
		} else {			
			MU::pause("error: Unknown action!");
			return 1;
		}			
    }
    catch (...) {
        cerr << endl << "Abnormal exception." << endl;
    }

	MU::pause();
    return 0;
}

/*
		//______ inc-fb2 - train ______
		action = "train";
		data = root + "130203-balls/2_processed_fann/data_train.txt";		
		name = root + "130203-balls/3_nets/net100inc-fb2";		
		alg = "incremental"; 
		hidden = 100;
		maxit = 1000;
		alfa = 0.4;
		err = 0.0005;	
		with_weight_factors = true;

		//______ inc-fb2 - test ______
		action= "test";
		data = root + "130203-balls/2_processed_fann/data_test.txt";		
		output = root + "130203-balls/3_nets/net100inc-fb2.csv";
		name = root + "130203-balls/3_nets/net100inc-fb2.net";		
		config = root + "130203-balls/2_processed_fann/data_parameters.txt";		
		positions = root + "130203-balls/2_processed_fann/data_test_positions.txt";				
		sizes = root + "130203-balls/2_processed_fann/data_test_sizes.txt";		
		eyes = root + "130203-balls/2_processed_fann/data_test_eyes.txt";								
		with_weight_factors = true;
		

		//______ rprop-fb - train ______
		action = "train";
		data = root + "130203-balls/2_processed_fann/data_train.txt";		
		name = root + "130203-balls/3_nets/net100inc-fb3";		
		alg = "incremental"; 
		hidden = 100;
		maxit = 1000;
		alfa = 0.4;
		err = 0.0003;	
		with_weight_factors = true;

		//______ rprop-fb - test ______
		action = "test";
		name = root + "130203-balls/3_nets/net100inc-fb3.net";			
		data = root + "130203-balls/2_processed/data_test.txt";				
		config = root + "130203-balls/2_processed/data_parameters.txt";		
		output = root + "130203-balls/3_nets/net100inc-fb3.csv";
		sizes = root + "130203-balls/2_processed/data_test_sizes.txt";		
		eyes = root + "130203-balls/2_processed/data_test_eyes.txt";
		with_weight_factors = true;
					
		action = "train";
		data = root + "130318/2_processed/19_train.txt";		
		config = root + "130318/2_processed/19_parameters.txt";	
		name = root + "130318/3_nets/19_net100inc-fb";		
		alg = "incremental"; 
		hidden = 100;
		maxit = 1000;
		alfa = 0.4;
		err = 0.0005;	
		with_weight_factors = true;
	
		action = "test";
		name = root + "130318/3_nets/19_net100inc-fb.net";			
		data = root + "130318/2_processed/19_test.txt";				
		config = root + "130318/2_processed/19_parameters.txt";		
		output = root + "130318/3_nets/19_net100inc-fb.csv";
		sizes = root + "130318/2_processed/19_test_sizes.txt";		
		eyes = root + "130318/2_processed/19_test_eyes.txt";
		with_weight_factors = true;
	

		action = "train";
		data = root + "130318/2_processed/10_train.txt";		
		config = root + "130318/2_processed/10_parameters.txt";	
		name = root + "130318/3_nets/10_net100inc-fb";		
		alg = "incremental"; 
		hidden = 100;
		maxit = 1000;
		alfa = 0.4;
		err = 0.0005;	
		with_weight_factors = true;

		action = "test";
		name = root + "130318/3_nets/10_net100inc-fb.net";			
		data = root + "130318/2_processed/10_test.txt";				
		config = root + "130318/2_processed/10_parameters.txt";		
		output = root + "130318/3_nets/10_net100inc-fb.csv";
		sizes = root + "130318/2_processed/10_test_sizes.txt";		
		eyes = root + "130318/2_processed/10_test_eyes.txt";
		with_weight_factors = true;
	
		
		action = "train";
		datasetName = "37";
		data = root + "130318/2_processed/" + datasetName +"_train.txt";		
		config = root + "130318/2_processed/" + datasetName +"_parameters.txt";	
		name = root + "130318/3_nets/" + datasetName + "_net100inc-fb";		
		alg = "incremental"; 
		hidden = 100;
		maxit = 1000;
		alfa = 0.4;
		err = 0.0005;	
		with_weight_factors = true;

		action = "test";
		name = root + "130318/3_nets/"+datasetName+"_net100inc-fb.net";			
		data = root + "130318/2_processed/"+datasetName+"_test.txt";				
		config = root + "130318/2_processed/" + datasetName +"_parameters.txt";					
		output = root + "130318/3_nets/"+datasetName+"_net100inc-fb.csv";
		sizes = root + "130318/2_processed/"+datasetName+"_test_sizes.txt";		
		eyes = root + "130318/2_processed/"+datasetName+"_test_eyes.txt";
		with_weight_factors = true;

		
		action = "train";
		datasetName = "19";
		data = root + "130318/2_processed/" + datasetName +"_train.txt";		
		config = root + "130318/2_processed/" + datasetName +"_parameters.txt";	
		name = root + "130318/3_nets/" + datasetName + "_net100incm-fb";		
		alg = "incremental"; 
		hidden = 100;
		maxit = 1000;
		alfa = 0.4;
		err = 0.0005;	
		momentum = 0.05;
		with_weight_factors = true;
		
		
		action = "test";
		name = root + "130318/3_nets/"+datasetName+"_net100incm-fb.net";			
		data = root + "130318/2_processed/"+datasetName+"_test.txt";				
		config = root + "130318/2_processed/" + datasetName +"_parameters.txt";					
		output = root + "130318/3_nets/"+datasetName+"_net100incm-fb.csv";
		sizes = root + "130318/2_processed/"+datasetName+"_test_sizes.txt";		
		eyes = root + "130318/2_processed/"+datasetName+"_test_eyes.txt";
		with_weight_factors = true;
		
*/

		/*
		with_weight_factors = false;
		action = "test";
						
		name = root + "130203-balls/3_nets/net100inc.net";

		data = root + "130203-balls/2_processed_fann/data_test.txt";		
		config = root + "130203-balls/2_processed_fann/data_parameters.txt";		

		positions = root + "130203-balls/2_processed_fann/data_test_positions.txt";				
		sizes = root + "130203-balls/2_processed_fann/data_test_sizes.txt";		
		eyes = root + "130203-balls/2_processed_fann/data_test_eyes.txt";
		
		output = root + "130203-balls/3_nets/net100inc.csv";
		*/

		/*	
		action = "test";
						
		name = root + "130203-balls/3_nets/net100inc-fb.net";

		data = root + "130203-balls/2_processed_fann/data_test.txt";		
		config = root + "130203-balls/2_processed_fann/data_parameters.txt";		

		positions = root + "130203-balls/2_processed_fann/data_test_positions.txt";				
		sizes = root + "130203-balls/2_processed_fann/data_test_sizes.txt";		
		eyes = root + "130203-balls/2_processed_fann/data_test_eyes.txt";
		
		output = root + "130203-balls/3_nets/net100inc-fb.csv";
		*/