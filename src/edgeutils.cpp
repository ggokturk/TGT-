#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <filesystem>
using namespace std;

std::vector<std::filesystem::path> get_directories(const std::string& s) {
	std::vector<std::filesystem::path> r;
	for (auto& p : std::filesystem::recursive_directory_iterator(s))
		if (p.is_directory())
			r.push_back(p.path());
	return r;
}

void graphtobin(string filename, string outfilename, bool directed = true, int skip = 0) {
	std::ifstream in(filename, std::fstream::in);
	in.sync_with_stdio(false);
	std::vector<std::vector<uint32_t>> adjlist;
	uint32_t s, v;
	size_t n = 0, m = 0;
	std::string line;
	while (std::getline(in, line)) {
		if (line.length() > 0 && (line.at(0) == '#' || line.at(0) == '%'))
			continue;
		if (skip > 0) {
			skip--;
			continue;
		}
		std::stringstream ss(line);

		ss >> s >> v;

		if (adjlist.size() < ((std::max)(s, v) + 1)) {
			adjlist.resize((std::max)(s, v) + 1);
		}

		adjlist[s].push_back(v);

		if (!directed)
			adjlist[v].push_back(s);
		n = (std::max)(size_t((std::max)(s, v) + 1), n);
	}
	for (size_t i = 0; i < adjlist.size(); i++) {
		m += adjlist[i].size();
	}
#pragma omp parallel for
	for (size_t i = 0; i < adjlist.size(); i++) {
		sort(adjlist[i].begin(), adjlist[i].end());
	}
	std::ofstream out(outfilename, std::fstream::out | std::fstream::binary);
	out.write((char*)&n, sizeof(size_t));
	out.write((char*)&m, sizeof(size_t));
	size_t pos = 0;
	out.write((char*)&pos, sizeof(size_t));
	for (size_t i = 0; i < n; i++) {
		pos += adjlist[i].size();
		out.write((char*)&pos, sizeof(size_t));
	}
	pos = 0;
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < adjlist[i].size(); j++) {
			uint32_t target = adjlist[i][j];
			out.write((char*)&target, sizeof(target)); ;
		}
	}
}
void eigentobin(string filename, string outfilename, int omega) {
	std::ifstream in(filename, ios::in);
	std::ofstream out(outfilename, ios::out|ios::binary);
	in.sync_with_stdio(false);
	string line;

	while(getline(in, line)){		
		float val = 0;
		stringstream ss(line);
		ss >> val;
		out.write((char*) & val, sizeof(float));
	}
	in.seekg(0);
	while (getline(in, line)) {
		double val = 0;
		stringstream ss(line);
		ss >> val;
		while (ss >> val) {
			ss >> val;
			out.write((char*)&val, sizeof(float));
		}
	}
}

int main(int argc, char** argv) {
	string method = string(argv[1]);
	if (method == "-g") {
		graphtobin(string(argv[2]), string(argv[3]), atoi(argv[4]), atoi(argv[5]));
	}
	else if (method == "-e") {
		eigentobin(string(argv[2]), string(argv[3]), atoi(argv[4]));
	}
	else if (method == "-gb") {
		string path = argv[1];		
		for (auto dir : get_directories(path)) {
			auto graphpath = dir / std::filesystem::path("graph.txt");		
			graphtobin(graphpath.string(), dir.filename().string() + ".bin", 0, 0);
		}
	}
} 