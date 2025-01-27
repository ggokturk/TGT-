#include "../vendor/aesc/algo.h"
#include "aesc.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>
using namespace std;

int
main()
{
  cout << fixed << setprecision(4);
  vector<string> datasets = {
    // "Epinions", "Gnutella", "EU",
    // "Enron",    "Facebook", "Twitch",
    // "HepPh",    "HepTh",
    // "Slashdot"
    "Orkut"
  };
  //"DBLP",/* "Amazon", "Youtube"*/};
  vector<float> eps = { 0.05
  , 0.02, 0.01, 0.005
  };
  vector<int> threads = { 16 
  , 8, 4, 2, 1
  };
  bool run_theirs = false, run_ours = true;

  for (int i = 0; i < datasets.size(); i++) {
    const auto& dataset = datasets[i];
    auto directory = filesystem::path(".") / filesystem::path("datasets");
    AESCConfig config;
    config.strFolder = directory.string();
    config.strGraph = dataset;

    config.omega = 128;
    config.gamma = 10;
    string path = config.strFolder + "/" + config.strGraph + "/graph.txt";
    string eigenpath = config.strFolder + "/" + config.strGraph +
                       "/sorted_eigens_" + std::to_string(config.omega) +
                       ".txt";

    AESCGraph g(path, eigenpath, config.omega);

    config.delta = 1.0 / g.n;
    for (int j = 0; j < eps.size(); j++) {
      const auto& e = eps[j];
      string outfile = "results/" + dataset + "_" + to_string(e);
      config.epsilon = e;
      bool first = true;
      if (run_ours) {
        for (const auto& t : threads) {
          omp_set_num_threads(t);

          AESC algo(g, config);
          double itime, ftime, exec_time;
          itime = omp_get_wtime();

          float* pred_secs = algo.tgtp();

          ftime = omp_get_wtime();

          exec_time = ftime - itime;
          auto total = accumulate(pred_secs, pred_secs + g.m, 0.0f);

          cout << std::setw(10) << "Ours" << std::setw(10) << dataset << "\t"
               << std::setw(10) << e << "\t" << std::setw(10) << t << "\t"
               << std::setw(10) << exec_time << "\t" << total * 0 << endl;
          // if (first) {
          //	first = false;
          //	std::ofstream out;
          //	out.open(outfile, ios::out | ios::binary);
          //	out.write((char*)pred_secs, g.m * sizeof(float));
          // }

          delete[] pred_secs;
        }
      }
      if (run_theirs) {
        Config konfig;
        konfig.strFolder = config.strFolder;
        konfig.strGraph = config.strGraph;

        konfig.epsilon = e;
        konfig.omega = 128;
        konfig.gamma = 10;

        Graph graph(konfig.strFolder, konfig.strGraph);
        konfig.delta = 1.0 / graph.getN();
        vector<ipair> seeds;
        vector<double> exact_secs;
        vector<double> errors;
        vector<pair<double, vector<double>>> Eigens;
        map<ipair, int> taus;
        double itime, ftime, exec_time;
        itime = omp_get_wtime();
        loadEigens(
          konfig.strFolder, konfig.strGraph, graph, Eigens, konfig.omega);
        calTau(Eigens, taus, graph, konfig.omega, konfig.epsilon / 2.0);
        map<ipair, double> pred_secs;
        truncatedGraphTravPlus(graph, pred_secs, taus, konfig);
        ftime = omp_get_wtime();
        exec_time = ftime - itime;
        float total = 0;
        for (auto it = pred_secs.cbegin(); it != pred_secs.cend(); ++it) {
          total += it->second;
        }

        cout << std::setw(10) << "Theirs" << std::setw(10) << dataset << "\t"
             << std::setw(10) << e << "\t" << std::setw(10) << 1 << "\t"
             << std::setw(10) << exec_time << "\t" << total * 0 << endl;
        std::ofstream out;
        // string theirfile = "results_theirs/" + dataset + "_" + to_string(e);
        // out.open(theirfile, ios::out | ios::binary);
        // for (auto it = pred_secs.cbegin(); it != pred_secs.cend(); ++it)
        //{
        //	out << it->first.first << "\t" << it->first.second << "\t" <<
        // it->second << "\n";
        // }
      }
    }
  }
}