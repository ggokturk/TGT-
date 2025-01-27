
#include <iostream>
#include <random>
#include <string>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <stdint.h>
#include "aesc.h"
#include "graph.h"
#include "algo.h"
using namespace std;



int main(int argc, char** argv) {
	fuser::AESCConfig config;
	config.strFolder = argv[1];
	config.strGraph = argv[2];

	config.epsilon = atof(argv[3]);
	config.omega = atof(argv[4]);
	config.gamma = atoi(argv[5]);
	string outfile = argv[6];
	string path = config.strFolder + "/" + config.strGraph + "/graph.txt";
	string eigenpath = config.strFolder + "/" + config.strGraph + "/sorted_eigens_" + std::to_string(config.omega) + ".txt";

	fuser::AESCGraph<float> g(path, eigenpath, config.omega);
	config.delta = 1.0 / g.n;
	AESC<float> algo(g, config);
	float* pred_secs = algo.tgtp();

	std::ofstream out;
	out.open(outfile, ios::out | ios::binary);
	out.write((char*)pred_secs, g.m * sizeof(float));
	delete[] pred_secs;
	return 0;
}