#pragma once
#include <string>
#include <algorithm>
#include <map>
#include <immintrin.h>
#include <cstdint>
#include <queue>
#include <utility>
#include <sstream>
#include <random>
#include <cfloat>
#include <climits>
#include <iostream>
#include "graph.h"
#include <unordered_set>
#include <execution>

using namespace std;

std::mt19937 rndgen;
#define RWRATIO 25
#define ITERLIMIT 10
#include <fstream>

namespace fuser {
	__m256i murmur(__m256i h) {
		auto h2 = _mm256_srli_epi32(h, 16);
		h = _mm256_xor_si256(h, h2);
		h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0x85ebca6b));
		h2 = _mm256_srli_epi32(h, 13);
		h = _mm256_xor_si256(h ,h2);
		h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0xc2b2ae35));
		h2 = _mm256_srli_epi32(h, 16);
		h = _mm256_xor_si256(h, h2);
		h = _mm256_and_si256(h, _mm256_set1_epi32(INT_MAX));
		return h;
	}
	__m256i avx2hash(__m256i h) {
		h =_mm256_mullo_epi32(h, _mm256_set1_epi32(0x01000193));
		h = _mm256_add_epi32(h, _mm256_set1_epi32(0x811c9dc5));
		h = _mm256_and_si256(h, _mm256_set1_epi32(INT_MAX));
		return h;
	}
}
template <typename T>
struct EigenVecs {
	EigenVecs() {};
	EigenVecs(size_t n, size_t omega) :
		len(n), omega(omega) {
		values = new T[omega];
		vectors = new T[omega * n];
	}
	void load_txt(const string& path) {
		std::ifstream infile(path);
		infile.sync_with_stdio(false);
		for (int i = 0; i < omega; ++i) {
			double val = 0;
			infile >> values[i];
			for (size_t j = 0; j < len; j++) {
				infile >> vectors[(omega * j) + i];
			}
		}
	}
	void load_bin(const string& path) {
		std::ifstream in(path);
		in.read(reinterpret_cast<char*>(values), sizeof(T) * omega);
		in.read(reinterpret_cast<char*>(vectors), sizeof(T) * omega * len);
	}
	T* begin(size_t i) {
		return &vectors[i * omega];
	}
	T* end(size_t i) {
		return &vectors[(i + 1) * omega];
	}

	T& operator[](size_t i) {
		return values[i];
	}
	T* values;
	T* vectors;
	size_t len, omega;
	void free() {
		if (values != NULL)
			delete[] values;
		if (vectors != NULL)
			delete[] vectors;
	}
};

struct AESCGraph : fuser::graph_t<uint32_t, uint32_t> {
	EigenVecs<float> eigens;
	AESCGraph(const string& graph_path, const string& eigen_path, size_t omega) {
		this->load_txt(graph_path);
		this->eigens = EigenVecs<float>(this->n, omega);
		this->eigens.load_txt(eigen_path);
	}
	void free() {
		graph_t::free();
		eigens.free();
	}
};
struct AESCConfig {
	double epsilon, delta, lambda;
	string strFolder, strGraph;
	int omega, gamma;
};

union alignas(32) avx2_t {
	__m256 f;
	__m256i i;
	float fs[8];
	uint32_t is[8];
	avx2_t() = default;
	avx2_t(__m256i rhs) :i(rhs) {}
	avx2_t(__m256 rhs) :f(rhs) {}
	avx2_t(int rhs) :i(_mm256_set1_epi32(rhs)) {}
	template <typename T>
	inline avx2_t(T* rhs) : i((__m256i)_mm256_load_si256(rhs)) {}
	inline static avx2_t iota() {
		return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
	}
	inline avx2_t operator^(avx2_t val) {
		return _mm256_xor_si256(i, val.i);
	}
	inline avx2_t operator&(avx2_t val) {
		return _mm256_and_si256(i, val.i);
	}
	inline avx2_t operator>>(int val) {
		return _mm256_srai_epi32(i, val);
	}
	inline avx2_t operator<<(int val) {
		return _mm256_slli_epi32(i, val);
	}
	inline avx2_t operator+(avx2_t val) {
		return _mm256_add_epi32(i, val.i);
	}
	inline avx2_t operator*(avx2_t val) {
		return _mm256_mul_epi32(i, val.i);
	}
	//inline avx2_t fmadd(avx2_t a, avx2_t b)(
	//	return _mm256_madd_ps(a.f, b.f, f);
	//)
	// inline avx2_t operator%(avx2_t val) {
	// 	return fuser::_mm256_rem_epu32(i, val.i);
	// }

	inline avx2_t hash() {
		avx2_t c2 = 0x27d4eb2d;
		avx2_t key = i;
		key = (key ^ 61) ^ (key >> 16);
		key = key + (key << 3);
		key = key ^ (key >> 4);
		key = key * c2;
		key = key ^ (key >> 15);
		key = (key & INT_MAX);
		return key;
	}
	inline avx2_t murmur() {
		avx2_t h = *this;
		h = h ^ (h >> 16);
		h = h * 0x85ebca6b;
		h = h ^ (h >> 13);
		h = h * 0xc2b2ae35;
		h = h ^ (h >> 16);
		h = (h & INT_MAX);
		return h;
	}
};


struct AESC {


	AESCGraph& g;
	AESCConfig& config;

	AESC(AESCGraph& g, AESCConfig& config) :
		g(g), config(config) {
	}
	float* tgtp() {
		float* pred_secs = new float[g.m];
		int* taus = calculate_taus(config.epsilon / 2.0);
		tgtp_baseline(pred_secs, taus);

		delete[] taus;
		return pred_secs;
	}
	float* mc() {
		float* pred_secs = new float[g.m];
	
		int* taus = calculate_taus(config.epsilon / 2.0);
		mc_baseline(pred_secs, taus);

		delete[] taus;
		return pred_secs;
	}


	int get_tau(size_t dv, size_t du, const float epsilon, const float lamba, const float delta, const float upsilon) {
		float eps_delta = epsilon - delta > 0.0 ? epsilon - delta : epsilon;
		float a = (std::log)((std::max)((1.0 / du + 1.0 / dv - 2.0 / du / dv - upsilon) / eps_delta / (1.0 - (lamba * lamba)), 1.0));
		float b = (std::log)((float)(1.0) / abs(lamba));
		int tau = (std::max)((std::ceil)((float)(a / b - 1)), (float)1.0);
		return tau % 2 == 0 ? int(tau + 1) : int(tau);
	}

	float get_delta(size_t u, size_t  v, int t, float* eigen_t_1) {
		float delta = 0;
		const int omega = g.eigens.omega;
		const auto* uf = g.eigens.begin(u);
		const auto* vf = g.eigens.begin(v);

		for (int i = 1; i < omega - 1; ++i) {
			const auto val = g.eigens[i];
			const auto diff = (uf[i] - vf[i]);
			const float eigen_t_1_i = eigen_t_1[i];
			delta += (diff * diff) * eigen_t_1_i / (1 - val);
		}
		return delta / g.m;
	}
	float get_upsilon(size_t u, size_t v) {
		float upsilon = 0;
		const int omega = g.eigens.omega;
		const auto* uf = g.eigens.begin(u);
		const auto* vf = g.eigens.begin(v);
		for (int i = 1; i < omega - 1; ++i) {
			const auto val = g.eigens[i];
			const auto diff = (uf[i] - vf[i]);
			upsilon += (diff * diff) * (1 + val);
		}
		return upsilon / g.m;
	}


	int* calculate_taus(const float epsilon) {

		int* taus = new int[g.m];
		std::fill(taus, taus + g.m, 0);
		const auto omega = g.eigens.omega;

#pragma omp parallel
		{
			float* __restrict eigen_t_1 = new float[omega];
			float* __restrict eigen_sq = new float[omega];
			for (size_t i = 0; i < omega; i++) {
				eigen_sq[i] = (g.eigens[i] * g.eigens[i]);
			}
#pragma omp for schedule(dynamic,100)
			for (long long u = 0; u < g.n; u++) {
				for (auto* it = g.begin(u); it < g.end(u); it++) {
					const auto v = *it;
					if (v < u) continue;
					float delta = 0;
					float upsilon = get_upsilon(u, v);

					float lamba = g.eigens[1];
					const size_t du = g.degree(u), dv = g.degree(v);
					int tau = get_tau(du, dv, epsilon, lamba, delta, upsilon);
					int t = 1;
					std::copy(&eigen_sq[0], &eigen_sq[omega], eigen_t_1);
					while (true) {
						delta = get_delta(u, v, t, eigen_t_1);
						lamba = g.eigens[g.eigens.omega - 1];
						int tauprime = get_tau(du, dv, epsilon, lamba, delta, upsilon);

						if (t <= tauprime && tauprime < tau) {
							tau = tauprime;
							t += 2;
						}
						else
							break;

						for (size_t i = 0; i < omega; i++) {
							eigen_t_1[i] *= eigen_sq[i];
						}
					}

					taus[(it - g.begin(0))] = tau;
					auto lower = std::lower_bound(g.begin(v), g.end(v), u);
					auto dist = distance(g.begin(0), lower);
					taus[dist] = tau;
				}
			}
			delete[] eigen_sq;
			delete[] eigen_t_1;
		}
		return taus;
	}

	//__declspec(noinline) 
	void diffuse(
			const size_t src,
			float* pvec,
			const size_t* q,
			const size_t q_size,
			size_t* q_next,
			size_t& q_next_size,
			const float ds,
			float* preds, char* visited) {
// #pragma omp parallel for
		for (int64_t i = 0; i < q_size; i++) {
			const auto v = q[i];
			float residue = pvec[v];
			for (auto* ptr = g.begin(v); ptr < g.end(v); ptr++) {
				const auto u = *ptr;
				if (!visited[u]) {
					q_next[q_next_size++] = u;
					visited[u] = true;
				}
				pvec[u] += residue / (float)g.degree(u);

			}
			pvec[v] = 0;
		}

		for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
			const auto vj = g.adj[pos];
			float update = (pvec[src] - pvec[vj]) / ds;
			preds[pos] += update;
		}
		for (size_t i = 0; i < q_next_size; i++) {
			const auto v = q_next[i];
			visited[v] = false;
		}
	}
	inline float reduce_fs(__m256 f) {
		const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(f, 1), _mm256_castps256_ps128(f));
		const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
		const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
		return _mm_cvtss_f32(x32);
	}

	////__declspec(noinline) 
	void pull_single(
			const size_t src,
			float* pvec,
			float* pvec_next,
			float* preds
		) {
#pragma omp parallel for
		for (int64_t u = 0; u < g.n; u++) {
			float agg = 0;
			float inv_ds = 1.0f / g.degree(u);
			for (auto* ptr = g.begin(u); ptr < g.end(u); ptr++) {
				const auto v = *ptr;
				agg += pvec[v] * inv_ds;
			}
			pvec_next[u] = agg;
		}
		const float ds = 1.0f / g.degree(src);
		for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
			const auto vj = g.adj[pos];
			float update = (pvec_next[src] - pvec_next[vj]) * ds;
			preds[pos] += update;
		}
	}
	//__declspec(noinline) 
	void pull_single_avx2(
			const avx2_t srcs,
			avx2_t* pvec,
			avx2_t* pvec_next,
			float* preds
		) {

		for (size_t u = 0; u < g.n; u++) {
			avx2_t agg = _mm256_setzero_ps();
			const auto inv_ds = _mm256_set1_ps(1.0f / g.degree(u));
			for (auto* ptr = g.begin(u); ptr < g.end(u); ptr++) {
				const auto v = *ptr;
				agg.f = _mm256_add_ps(_mm256_mul_ps(pvec[v].f, inv_ds), agg.f);
			}
			pvec_next[u] = agg;
		}
		for (int b = 0; b < 8; b++) {
			size_t src = srcs.is[b];
			const float ds = 1.0f / g.degree(src);
			for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
				const auto vj = g.adj[pos];
				float update = (pvec_next[src].fs[b] - pvec_next[vj].fs[b]) * ds;
				preds[pos] += update;
			}
		}
	}



	//__declspec(noinline) 
	float randomwalk_2way(size_t src, size_t v, const size_t num_walk, const size_t len_walk, float* pvec) {
		float x = 0;
		for (size_t i = 0; i < num_walk; i++) {
			size_t cur = src;
			size_t cur2 = v;
			for (size_t len = 0; len < len_walk; len++) {
				const size_t k = rndgen() % g.degree(cur);
				cur = g.begin(cur)[k];
				const size_t k2 = rndgen() % g.degree(cur2);
				cur2 = g.begin(cur2)[k2];
				x += (pvec[cur] - pvec[cur2]);
			}
		}
		return x;
	}
	//__declspec(noinline) 
	float randomwalk_2way_avx2(
			uint32_t src,
			uint32_t v,
			const uint32_t num_walk,
			const uint32_t len_walk,
			float* pvec) {
		const int BLOCKSIZE = 8;

		avx2_t x;
		x.f = _mm256_setzero_ps();

		for (uint32_t i = 0; i < num_walk; i += BLOCKSIZE) {
			avx2_t cur, cur2;
			cur.i = _mm256_set1_epi32(src);
			cur2.i = _mm256_set1_epi32(v);

			//rnds[i] = (i * (size_t)UINT32_MAX) / n;
			__m256i secret = _mm256_cvtps_epi32( // FIXME: _EPU32 PRODUCES ILLEGAL INSTRUCTION 
				_mm256_mul_ps(
					_mm256_add_ps(
						_mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f),
						_mm256_set1_ps(float(i))
					),
					_mm256_set1_ps(float(INT32_MAX) / num_walk)
				)
			);

			for (uint32_t len = 0; len < len_walk; len++) {
				avx2_t rnd, ds, rnd2, ds2;
				
				//rnd = secret ^ cur.hash();
				//rnd2 = secret ^ cur2.hash();
				rnd = _mm256_xor_si256(secret, fuser::murmur(cur.i));
				rnd2 = _mm256_xor_si256(secret, fuser::murmur(cur2.i));
				ds = _mm256_set_epi32(
					g.degree(cur.is[7]), g.degree(cur.is[6]),
					g.degree(cur.is[5]), g.degree(cur.is[4]),
					g.degree(cur.is[3]), g.degree(cur.is[2]),
					g.degree(cur.is[1]), g.degree(cur.is[0]));
				ds2 = _mm256_set_epi32(
					g.degree(cur2.is[7]), g.degree(cur2.is[6]),
					g.degree(cur2.is[5]), g.degree(cur2.is[4]),
					g.degree(cur2.is[3]), g.degree(cur2.is[2]),
					g.degree(cur2.is[1]), g.degree(cur2.is[0]));

				//avx2_t k = rnd % ds, k2 = rnd2 % ds2;
				avx2_t k, k2;

				k.i = _mm256_rem_epu32(rnd.i, ds.i);
				k2.i = _mm256_rem_epu32(rnd2.i, ds2.i);

				cur = _mm256_set_epi32(
					g.begin(cur.is[7])[k.is[7]], g.begin(cur.is[6])[k.is[6]],
					g.begin(cur.is[5])[k.is[5]], g.begin(cur.is[4])[k.is[4]],
					g.begin(cur.is[3])[k.is[3]], g.begin(cur.is[2])[k.is[2]],
					g.begin(cur.is[1])[k.is[1]], g.begin(cur.is[0])[k.is[0]]
				);
				cur2 = _mm256_set_epi32(
					g.begin(cur2.is[7])[k2.is[7]], g.begin(cur2.is[6])[k2.is[6]],
					g.begin(cur2.is[5])[k2.is[5]], g.begin(cur2.is[4])[k2.is[4]],
					g.begin(cur2.is[3])[k2.is[3]], g.begin(cur2.is[2])[k2.is[2]],
					g.begin(cur2.is[1])[k2.is[1]], g.begin(cur2.is[0])[k2.is[0]]
				);
				x.f = _mm256_add_ps(x.f,
					_mm256_sub_ps(
						_mm256_set_ps(
							pvec[cur.is[7]], pvec[cur.is[6]],
							pvec[cur.is[5]], pvec[cur.is[4]],
							pvec[cur.is[3]], pvec[cur.is[2]],
							pvec[cur.is[1]], pvec[cur.is[0]]
						),
						_mm256_set_ps(
							pvec[cur2.is[7]], pvec[cur2.is[6]],
							pvec[cur2.is[5]], pvec[cur2.is[4]],
							pvec[cur2.is[3]], pvec[cur2.is[2]],
							pvec[cur2.is[1]], pvec[cur2.is[0]]
						)
					)
				);
			}
		}
		float sum = 0;
		for (int b = 0; b < BLOCKSIZE; b++) {
			sum += x.fs[b];
		}
		return sum;
	}

	float get_chi(const size_t vi, const size_t vj, float* pvec, int len_walk, float& global_min, float& global_max, float& edge_max) {

		float src_local_max = 0;
		float src_local_min = FLT_MAX;
		for (auto* ptr = g.begin(vi); ptr < g.end(vi); ptr++) {
			const auto val = *ptr;
			src_local_max = (std::max)(src_local_max, pvec[val]);
			src_local_min = (std::min)(src_local_min, pvec[val]);
		}

		float tgt_local_max = 0;
		float tgt_local_min = FLT_MAX;
		for (auto* ptr = g.begin(vj); ptr < g.end(vj); ptr++) {
			const auto val = *ptr;
			tgt_local_max = (std::max)(tgt_local_max, pvec[val]);
			tgt_local_min = (std::min)(tgt_local_min, pvec[val]);
		}

		float chi = global_max + (src_local_max + tgt_local_max) / 2.0 + (len_walk - 1) * edge_max - src_local_min - tgt_local_min - 2 * (len_walk - 1) * global_min;
		return chi;
	}
	float get_chi2(int b, const size_t vi, const size_t vj, avx2_t* pvec, int len_walk, float& global_min, float& global_max, float& edge_max) {

		float src_local_max = 0;
		float src_local_min = FLT_MAX;
		for (auto* ptr = g.begin(vi); ptr < g.end(vi); ptr++) {
			const auto val = *ptr;
			src_local_max = (std::max)(src_local_max, pvec[val].fs[b]);
			src_local_min = (std::min)(src_local_min, pvec[val].fs[b]);
		}

		float tgt_local_max = 0;
		float tgt_local_min = FLT_MAX;
		for (auto* ptr = g.begin(vj); ptr < g.end(vj); ptr++) {
			const auto val = *ptr;
			tgt_local_max = (std::max)(tgt_local_max, pvec[val].fs[b]);
			tgt_local_min = (std::min)(tgt_local_min, pvec[val].fs[b]);
		}

		float chi = global_max + (src_local_max + tgt_local_max) / 2.0 + (len_walk - 1) * edge_max - src_local_min - tgt_local_min - 2 * (len_walk - 1) * global_min;
		return chi;
	}


	float get_edge_max(size_t* q, size_t q_size, float* pvec, const float& global_max, int gamma) {
		int nnz_size = q_size;
		int real_gamma = min(gamma, nnz_size);
		//if (real_gamma == 0) return 0;
		nth_element(q, q + real_gamma - 1, q + q_size, [&](const size_t A, const size_t B) -> bool {
			return pvec[A] > pvec[B]; });

		float gamma_max = 1;
		// vector<int8_t> candidates_exist(g.n, false);
		unordered_set<size_t> set;
		for (int i = 0; i < real_gamma; ++i) {
			auto& node = q[i];
			auto& val = pvec[node];
			gamma_max = (std::min)(gamma_max, val);
			// candidates_exist[node] = true;
			set.insert(node);
		}

		gamma_max = gamma < nnz_size ? gamma_max : 0;
		float edge_max = global_max + gamma_max;

		for (int i = 0; i < real_gamma; ++i) {
			auto& u = q[i];
			for (auto* ptr = g.begin(u); ptr < g.end(u); ptr++) {
				const auto v = *ptr;
				if (!set.contains(v))//!candidates_exist[v])
					continue;
				edge_max = (std::max)(edge_max, pvec[u] + pvec[v]);
			}
		}
		return edge_max;
	}

	float get_edge_max(int b, uint32_t* q, uint32_t q_size, avx2_t* pvec, float& global_max, int gamma) {
		int nnz_size = q_size;
		int real_gamma = min(gamma, nnz_size);
		if (real_gamma == 0) return 0;
		nth_element(q, q + real_gamma - 1, q + q_size, [&](const uint32_t A, const uint32_t B) -> bool {
			return pvec[A].fs[b] > pvec[B].fs[b]; });

		float gamma_max = 1;
		vector<bool> candidates_exist(g.n, false);
		for (int i = 0; i < real_gamma; ++i) {
			auto& node = q[i];
			auto& val = pvec[node].fs[b];
			gamma_max = (std::min)(gamma_max, val);
			candidates_exist[node] = true;
		}

		gamma_max = gamma < nnz_size ? gamma_max : 0;
		float edge_max = global_max + gamma_max;

		for (int i = 0; i < real_gamma; ++i) {
			auto& u = q[i];
			for (auto* ptr = g.begin(u); ptr < g.end(u); ptr++) {
				const auto v = *ptr;
				if (!candidates_exist[v])
					continue;
				edge_max = (std::max)(edge_max, pvec[u].fs[b] + pvec[v].fs[b]);
			}
		}
		return edge_max;
	}


	void tgtp_baseline(float* preds, int* taus) {
		rndgen.seed(42);
		std::fill(preds, preds + g.m, 0);
		const int BLOCKSIZE = 8;

		int cnt = 0;
		float numrw = 0;
		int itermax= 0;

#pragma omp parallel
		{
			char* visited = new char[g.n];
			std::fill(visited, visited + g.n, 0);
			size_t* q = new size_t[g.n];
			size_t q_size = 0;
			size_t* q_next = new size_t[g.n];
			size_t q_next_size = 0;
			// unordered_set<float> pvec;
			// float* pvec = new float[g.n];
			//FIXME, FIND MAXIMUM FOR NUM_WALKS
#pragma omp for schedule(dynamic)
			for (int64_t src = 0; src < g.n; src++) {
				auto pvec_ptr = std::make_unique<float[]>(g.n);
				float* pvec = pvec_ptr.get();
				// for (size_t i=0; i<g.n; i++){
				// 	pvec[i]=0;
				// }
				int ell = 0;
				pvec[src] = 1.0;

				auto ds = (float)g.degree(src);
				const auto inv_ds = 1 / ds;

				q_size = 0;
				q_next_size = 0;
				q[q_size++] = src;
				for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
					const auto v = g.adj[pos];
					preds[pos] = 1.0 / ds;
				}
				float global_min, global_max;

				for(int iter=0; iter<ITERLIMIT; iter++){ //while (true) {
					global_min = 1;
					global_max = 0;
					float picost = 0, rwcost = 0;

					diffuse(src, pvec, q, q_size, q_next, q_next_size, ds, preds, visited);
					swap(q, q_next);
					q_size = q_next_size;
					q_next_size = 0;

					ell++;
					for (size_t i = 0; i < q_size; i++) {
						const auto v = q[i];
						picost += (float)g.degree(v);
						auto& val = pvec[v];
						global_max = max(global_max, val);
						global_min = min(global_min, val);
					}
					global_min = q_size < g.n ? 0 : global_min;

					for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
						const auto v = g.adj[pos];
						int rest_ell = taus[pos] - ell;
						if (rest_ell <= 0)
							continue;
						float chi = 2.0 * rest_ell * (global_max - global_min);
						float nr = max(ceil(8 * pow(chi, 2) * log(g.m / config.delta) / pow(ds * config.epsilon, 2)), 1.0);
						rwcost += nr;
					}
					if (picost >= RWRATIO * rwcost)
						break;
				}
				float edge_max = get_edge_max(q, q_size, pvec, global_max, config.gamma);
				if (config.gamma == 1)
					edge_max = 2 * global_max;

				for (int64_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
					const auto v = g.adj[pos];

					int64_t rest_ell = taus[pos] - ell;
					if (rest_ell <= 0)
						continue;

					float chi = get_chi(src, v, pvec, rest_ell, global_min, global_max, edge_max);
					if (config.gamma <= 0) {
						chi = 2 * rest_ell * global_max;
					}
					const float dse = ds * config.epsilon;
					volatile size_t nr = max(ceil(8 * (chi * chi) * log(g.m / config.delta) / (dse * dse)), 1.0);
					nr += (BLOCKSIZE - (nr % BLOCKSIZE)); //EXPLICITLY COMPLEMENTING TO THE NEXT MULTIPLE OF BLOCKSIZE
					numrw += nr;

					float xij = randomwalk_2way_avx2(src, v, nr, rest_ell, pvec);
#pragma omp atomic
					preds[pos] += xij / (ds * nr);
				}
			}
			delete[] q;
			delete[] q_next;
			delete[] visited;
			// delete[] pvec;
		}
		float* rev_preds = new float[g.m];
		#pragma omp parallel for
		for(int64_t i=0; i<g.m; i++){
			rev_preds[i]=0;
		}
		//std::fill(rev_preds, rev_preds + g.m, 0);
		#pragma omp parallel for
		for (size_t u = 0; u < g.n; u++) {
			for (size_t pos = g.idx[u]; pos < g.idx[u + 1]; pos++) {
				const auto v = g.adj[pos];
				auto* vu = std::lower_bound(g.begin(v), g.end(v), u);
				size_t vu_pos = distance(g.begin(0), vu);
				rev_preds[vu_pos] = preds[pos];
			}
		}
		#pragma omp parallel for
		for (size_t i = 0; i < g.m; i++)
			preds[i] += rev_preds[i];

		delete[] rev_preds;
	}
	void tgtp_pull(float* preds, int* taus) {
		const int BLOCKSIZE = 8;

		std::fill(preds, preds + g.m, 0);
		int cnt = 0;
		float numrw = 0;
#pragma omp parallel
		{
			char* visited = new char[g.n];
			std::fill(visited, visited + g.n, 0);
			size_t* q = new size_t[g.n];
			size_t q_size = 0;
			float* pvec = new float[g.n];
			float* pvec_next = new float[g.n];
			//FIXME, FIND MAXIMUM FOR NUM_WALKS
#pragma omp for
			for (int64_t src = 0; src < g.n; src++) {
				int ell = 0;
				std::fill(pvec, pvec + g.n, 0);
				std::fill(pvec_next, pvec_next + g.n, 0);
				pvec[src] = 1.0;

				auto ds = (float)g.degree(src);

				q_size = 0;
				for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
					const auto v = g.adj[pos];
					preds[pos] = 1.0 / ds;
				}
				float global_min, global_max;
				while (true) {
					global_min = 1;
					global_max = 0;
					float picost = 0, rwcost = 0;

					pull_single(src, pvec, pvec_next, preds);
					q_size = 0;
					for (size_t idx = 0; idx < g.n; idx++) {
						if (pvec_next[idx] != 0) {
							q[q_size++] = idx;
						}
					}
					swap(pvec, pvec_next);
					std::fill(pvec_next, pvec_next + g.n, 0);

					ell++;
					for (size_t i = 0; i < q_size; i++) {
						const auto v = q[i];
						picost += (float)g.degree(v);
						auto& val = pvec[v];
						global_max = max(global_max, val);
						global_min = min(global_min, val);
					}
					global_min = q_size < g.n ? 0 : global_min;

					for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
						const auto v = g.adj[pos];
						int rest_ell = taus[pos] - ell;
						if (rest_ell <= 0)
							continue;
						float chi = 2.0 * rest_ell * (global_max - global_min);
						float nr = max(ceil(8 * pow(chi, 2) * log(g.m / config.delta) / pow(ds * config.epsilon, 2)), 1.0);
						rwcost += nr;
					}
					if (picost >= RWRATIO * rwcost)
						break;
				}


				float edge_max = get_edge_max(q, q_size, pvec, global_max, config.gamma);
				if (config.gamma == 1)
					edge_max = 2 * global_max;
				for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
					const auto v = g.adj[pos];

					int64_t rest_ell = taus[pos] - ell;
					if (rest_ell <= 0)
						continue;

					float chi = get_chi(src, v, pvec, rest_ell, global_min, global_max, edge_max);
					if (config.gamma <= 0) {
						chi = 2 * rest_ell * global_max;
					}
					const float dse = ds * config.epsilon;
					size_t nr = max(ceil(8 * (chi * chi) * log(g.m / config.delta) / (dse * dse)), 1.0);
					nr += (BLOCKSIZE - (nr % BLOCKSIZE)); //EXPLICITLY COMPLEMENTING TO NEXT MULTIPLE OF BLOCKSIZE
					numrw += nr;
					float xij = randomwalk_2way_avx2(src, v, nr, rest_ell, pvec);
					preds[pos] += xij / (ds * nr);
				}
			}
			delete[] q;
			delete[] pvec;
			delete[] pvec_next;
		}
		float* rev_preds = new float[g.m];
		std::fill(rev_preds, rev_preds + g.m, 0);
		for (size_t u = 0; u < g.n; u++) {
			for (size_t pos = g.idx[u]; pos < g.idx[u + 1]; pos++) {
				const auto v = g.adj[pos];
				auto* vu = std::lower_bound(g.begin(v), g.end(v), u);
				size_t vu_pos = distance(g.begin(0), vu);
				rev_preds[vu_pos] = preds[pos];
			}

		}
		for (size_t i = 0; i < g.m; i++)
			preds[i] += rev_preds[i];
		//std::cout << "num_rw: " << numrw << std::endl;
		delete[] rev_preds;
	}
	template <typename T, typename T2>
	void parfill(T* ptr, T* end, T2 val) {
		int64_t size = end - ptr;
#pragma omp parallel for
		for (int64_t i=0; i< size; i++) {
			ptr[i] = val;
		}
	}
	//__declspec(noinline)
	void randomwalk_mc(uint32_t src, size_t len_walk, size_t n_walk, float* pvec) {
		for (uint32_t len = 0; len < len_walk; len++)
		{
			for (size_t i = 0; i < n_walk; i++) {
				uint32_t cur = src;
				for (uint32_t j = 0; j < len; j++) {
					uint32_t deg = g.degree(cur);
					uint32_t k = rndgen() % deg;
					cur = g.begin(cur)[k];
				}
				pvec[cur] += 1.0f;// / n_walk / g.degree(cur);
			}
		}
	}
	//__declspec(noinline)
	void randomwalk_mc_avx2(uint32_t src, size_t len_walk, size_t n_walk, float* pvec) {
		for (size_t len = 0; len < len_walk; len++) {
			const int BLOCKSIZE = 8;

			for (size_t i = 0; i < n_walk; i+=BLOCKSIZE) {
				avx2_t secret = _mm256_cvtps_epi32( // FIXME: _EPU32 PRODUCES ILLEGAL INSTRUCTION 
					_mm256_mul_ps(
						_mm256_add_ps(
							_mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f),
							_mm256_set1_ps(float(i))
						),
						_mm256_set1_ps(float(INT32_MAX) / n_walk)
					)
				);
				avx2_t cur;
				cur.i = _mm256_set1_epi32(src);



				for (uint32_t j = 0; j < len; j++) {
					avx2_t rnd, ds;
					rnd.i = _mm256_set_epi32(
							rndgen(), rndgen(), rndgen(), rndgen(),
							rndgen(), rndgen(), rndgen(), rndgen()
						);
					rnd = rnd & INT_MAX;
					//avx2_t roundhash = avx2_t(j).hash();
					//const auto jvec = _mm256_set1_epi32(len);
					//rnd = secret ^ (cur*jvec).hash();//^ roundhash;
					//const auto jvec = _mm256_set1_epi32(_mm_crc32_u32(src, len));
					//rnd = secret ^ jvec ^ (cur).hash();//^ roundhash;
					//rnd = secret ^ (cur).hash();//^ roundhash;
					ds = _mm256_set_epi32(
						g.degree(cur.is[7]), g.degree(cur.is[6]),
						g.degree(cur.is[5]), g.degree(cur.is[4]),
						g.degree(cur.is[3]), g.degree(cur.is[2]),
						g.degree(cur.is[1]), g.degree(cur.is[0]));
					avx2_t k;
					k.i = _mm256_rem_epu32(rnd.i, ds.i);
		
					//uint32_t deg = g.degree(cur);
					//uint32_t k = rndgen() % deg;
					//cur = g.begin(cur)[k];
					cur = _mm256_set_epi32(
						g.begin(cur.is[7])[k.is[7]], g.begin(cur.is[6])[k.is[6]],
						g.begin(cur.is[5])[k.is[5]], g.begin(cur.is[4])[k.is[4]],
						g.begin(cur.is[3])[k.is[3]], g.begin(cur.is[2])[k.is[2]],
						g.begin(cur.is[1])[k.is[1]], g.begin(cur.is[0])[k.is[0]]
					);
				}
				pvec[cur.is[0]] += 1.0f;
				pvec[cur.is[1]] += 1.0f;
				pvec[cur.is[2]] += 1.0f;
				pvec[cur.is[3]] += 1.0f;
				pvec[cur.is[4]] += 1.0f;
				pvec[cur.is[5]] += 1.0f;
				pvec[cur.is[6]] += 1.0f;
				pvec[cur.is[7]] += 1.0f;
			}
		}
	}
	void randomwalk_mc_alt(uint32_t src, size_t len_walk, size_t n_walk, float* pvec) {
		//for (uint32_t len = 0; len < len_walk; len++) {
		const size_t len = len_walk;
			const int BLOCKSIZE = 8;
			for (size_t i = 0; i < n_walk; i += BLOCKSIZE) {
				avx2_t secret = _mm256_cvtps_epi32( // FIXME: _EPU32 PRODUCES ILLEGAL INSTRUCTION 
					_mm256_mul_ps(
						_mm256_add_ps(
							_mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f),
							_mm256_set1_ps(float(i))
						),
						_mm256_set1_ps(float(INT32_MAX) / n_walk)
					)
				);
				avx2_t cur;
				cur.i = _mm256_set1_epi32(src);
				pvec[cur.is[0]] += 1.0f;
				pvec[cur.is[1]] += 1.0f;
				pvec[cur.is[2]] += 1.0f;
				pvec[cur.is[3]] += 1.0f;
				pvec[cur.is[4]] += 1.0f;
				pvec[cur.is[5]] += 1.0f;
				pvec[cur.is[6]] += 1.0f;
				pvec[cur.is[7]] += 1.0f;
				for (uint32_t j = 0; j < len; j++) {
					avx2_t rnd, ds;
					rnd = secret ^ cur.hash();
					ds = _mm256_set_epi32(
						g.degree(cur.is[7]), g.degree(cur.is[6]),
						g.degree(cur.is[5]), g.degree(cur.is[4]),
						g.degree(cur.is[3]), g.degree(cur.is[2]),
						g.degree(cur.is[1]), g.degree(cur.is[0]));
					avx2_t k;
					k.i = _mm256_rem_epu32(rnd.i, ds.i);

					//uint32_t deg = g.degree(cur);
					//uint32_t k = rndgen() % deg;
					//cur = g.begin(cur)[k];
					cur = _mm256_set_epi32(
						g.begin(cur.is[7])[k.is[7]], g.begin(cur.is[6])[k.is[6]],
						g.begin(cur.is[5])[k.is[5]], g.begin(cur.is[4])[k.is[4]],
						g.begin(cur.is[3])[k.is[3]], g.begin(cur.is[2])[k.is[2]],
						g.begin(cur.is[1])[k.is[1]], g.begin(cur.is[0])[k.is[0]]
					);
					pvec[cur.is[0]] += 1.0f;
					pvec[cur.is[1]] += 1.0f;
					pvec[cur.is[2]] += 1.0f;
					pvec[cur.is[3]] += 1.0f;
					pvec[cur.is[4]] += 1.0f;
					pvec[cur.is[5]] += 1.0f;
					pvec[cur.is[6]] += 1.0f;
					pvec[cur.is[7]] += 1.0f;
				}

			}
		//}
	}
	void mc_baseline(float* preds, int* taus) {
		fill(preds, preds + g.m, 0);
		vector<int> edgeid(g.m);
		iota(begin(edgeid), end(edgeid), 0);
		shuffle(edgeid.begin(), edgeid.end(), std::mt19937(std::random_device()()));

#pragma omp parallel
		{

			float* pvec = new float[g.n];
			fill(pvec, pvec + g.n, 0);
#pragma omp for schedule(dynamic)
			for (int64_t src = 0; src < g.n; src++) {
				size_t len_walk = 0;
				for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
					if (len_walk < taus[pos]) {
						len_walk = taus[pos];
					}
				}
				auto n_walk = uint64_t(40 * len_walk * log(/*8 * graph.getM() */ 4 * (g.m) * len_walk / config.delta) / config.epsilon / config.epsilon);
				const auto BLOCKSIZE = 8;
				n_walk += BLOCKSIZE - (n_walk % BLOCKSIZE);			
				if (n_walk >= INTMAX_MAX) {
					n_walk = INTMAX_MAX - (INTMAX_MAX % BLOCKSIZE);
				}
				randomwalk_mc_avx2(src, len_walk, n_walk, pvec);

				for (size_t pos = g.idx[src]; pos < g.idx[src + 1]; pos++) {
					uint32_t tgt = g.adj[pos];
					preds[pos] += pvec[src] / n_walk / g.degree(src) - pvec[tgt] / n_walk / g.degree(tgt);
					auto lower = std::lower_bound(g.begin(tgt), g.end(tgt), src);
					auto dist = distance(g.begin(0), lower);
					preds[dist] += pvec[src] / n_walk / g.degree(src) - pvec[tgt] / n_walk / g.degree(tgt);
				}
				fill(pvec, pvec + g.n, 0);
			}
			delete[] pvec;
		}
	}

};

map<pair<size_t, size_t>, float>
load_seed(const string& filename) {
	FILE* fin = fopen(filename.c_str(), "r");
	size_t s, t;
	float st;
	map<pair<size_t, size_t>, float> scores;

	while (fscanf(fin, "%zu %zu %f", &s, &t, &st) != EOF) {
		scores.emplace(std::make_pair(s, t), st);
	}
	fclose(fin);
	return scores;
}