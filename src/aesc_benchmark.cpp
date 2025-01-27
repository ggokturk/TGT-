#include "../vendor/aesc/algo.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>
using namespace std;

int main() {
  cout << fixed << setprecision(4);
  vector<string> datasets = {"Facebook", "Twitch", "HepPh", "Slashdot", "HepTh",
                             /*"DBLP",      "Amazon"*/};
  vector<float> eps = {0.05, 0.02, 0.01, 0.005};
  vector<int> threads = {16, 8, 4, 2, 1};
  bool run_theirs = true, run_ours = false;
#pragma omp parallel for collapse(2)
  for (int i = 0; i < datasets.size(); i++) {
    for (int j = 0; j < eps.size(); j++) {
      auto dataset = datasets[i];
      auto directory = filesystem::path("..") / filesystem::path("..") /
                       filesystem::path("datasets");

      Config konfig;
      konfig.strFolder = directory.string();
      konfig.strGraph = dataset;

      konfig.omega = 128;
      konfig.gamma = 10;

      Graph graph(konfig.strFolder, konfig.strGraph);
      konfig.delta = 1.0 / graph.getN();
      auto e = eps[j];
      konfig.epsilon = e;
      vector<ipair> seeds;
      vector<double> exact_secs;
      vector<double> errors;
      vector<pair<double, vector<double>>> Eigens;
      map<ipair, int> taus;
      double itime, ftime, exec_time;
      loadEigens(konfig.strFolder, konfig.strGraph, graph, Eigens,
                 konfig.omega);
      itime = omp_get_wtime();
      calTau(Eigens, taus, graph, konfig.omega, konfig.epsilon / 2.0);
      map<ipair, double> pred_secs;
      truncatedGraphTravPlus(graph, pred_secs, taus, konfig);
      ftime = omp_get_wtime();
      float total = 0;
      for (auto it = pred_secs.cbegin(); it != pred_secs.cend(); ++it) {
        total += it->second;
      }
      exec_time = ftime - itime;
#pragma omp critical
      {
        cout << std::setw(10) << "Theirs" << std::setw(10) << dataset << "\t"
             << std::setw(10) << e << "\t" << std::setw(10) << 1 << "\t"
             << std::setw(10) << exec_time << "s"
             << "\t" << total * 0 << endl;
      }
    }
  }
}
