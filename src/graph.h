#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>
namespace fuser {
template<typename idx_t=uint32_t, typename vert_t=uint32_t>
struct graph_t
{
  idx_t* __restrict idx = NULL;
  vert_t* __restrict adj = NULL;
  uint32_t* __restrict w = NULL;
  size_t n, m;
  vert_t degree(vert_t vertex)
  { // FIXME: graph should have a degree field, this only works with unweighted
    // graphs or edge duplication
    return end(vertex) - begin(vertex);
  }
  vert_t* begin(vert_t vertex) { return adj + idx[vertex]; }
  vert_t* end(vert_t vertex) { return adj + idx[vertex + 1]; }
  void free()
  {
    if (idx != NULL)
      delete[] idx;
    if (adj != NULL)
      delete[] adj;
    if (w != NULL)
      delete[] w;
  }
  graph_t& load_bin(const std::string& filename)
  {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in) {
      std::cerr << "IO error" << std::endl;
      exit(-1);
    }
    in.read(reinterpret_cast<char*>(&this->n), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&this->m), sizeof(size_t));
    this->idx = new idx_t[this->n + 1];
    this->adj = new vert_t[this->m];

    in.read(reinterpret_cast<char*>(this->idx), sizeof(idx_t) * (this->n + 1));
    in.read(reinterpret_cast<char*>(this->adj), sizeof(vert_t) * (this->m));
    return *this;
  }

  graph_t& load_txt(std::string filename,
                    bool directed = false,
                    size_t skip = 0)
  {
    std::ifstream in(filename, std::fstream::in);
    in.sync_with_stdio(false);
    std::vector<std::vector<vert_t>> adjlist;
    vert_t s, v;
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
    for (int64_t i = 0; i < adjlist.size(); i++) {
      sort(adjlist[i].begin(), adjlist[i].end());
    }

    this->n = n;
    this->m = m;
    this->idx = new idx_t[n + 1];
    this->adj = new vert_t[m];
    idx_t pos = 0;
    this->idx[0] = 0;
    for (size_t i = 0; i < n; i++) {
      pos += adjlist[i].size();
      this->idx[i + 1] = pos;
    }
    pos = 0;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < adjlist[i].size(); j++) {
        this->adj[pos++] = adjlist[i][j];
      }
    }
    return *this;
  }
};
}