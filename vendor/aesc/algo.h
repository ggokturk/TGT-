#pragma once
#include "graph.h"
int calTau(uint u, uint v, vector<pair<double,vector<double>>> &Eigens,const Graph& graph, int omega, double epsilon);
void calTau(vector<pair<double,vector<double>>> &Eigens, map<ipair,int>& Taus, const Graph& graph, int omega, double epsilon);
void push_single(uint src, const Graph &graph, vector<double> &pvec, vector<uint> &S, double ds, vector<double> &tmp_hvec);
void monteCarlo(uint src, int len_walk, uint64 n_walk, map<uint,double>& mapVals, const Graph& graph);
void truncatedGraphTrav(const Graph& graph, map<ipair,double>& preds, map<ipair,int> &taus);
double calEdgeMax(const Graph &graph, vector<uint> &S, vector<double> &pvec, const double &global_max, int gamma);
double calChi(uint vi,uint vj, const Graph& graph, vector<double> &pvec, vector<uint> &S, int len_walk, double &global_min, double &global_max, double &edge_max);
double RW_single(uint src, const Graph& graph, double num_walk, int len_walk, vector<double> &pvec);
double RW_double(uint src, uint v, const Graph& graph, double num_walk, int len_walk, vector<double> &pvec);
void truncatedGraphTravPlus(const Graph& graph, map<ipair,double>& preds, map<ipair,int> &taus, Config &config);
void monteCarlo_C(uint src, int len_walk, double eps, map<uint,double>& mapVals, const Graph& graph);
void loadSeed(const string& folder, const string& file_name, const Graph& graph, vector<ipair>& vecSeeds, vector<double>& vecER);
void loadEigens(const string& folder, const string& file_name, const Graph& graph, vector<pair<double, vector<double>>>& Eigens, int omega);