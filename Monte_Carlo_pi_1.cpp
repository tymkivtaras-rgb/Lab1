#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <immintrin.h>

using namespace std;

// Швидкий xorshift 
inline uint64_t xorshift64(uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// Генерація double з uint64 
inline double fast_double(uint64_t& state) {
    return (xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
}

int main() {

    vector<unsigned long long> N_values = {
        1'000'000ULL,
        10'000'000ULL,
        100'000'000ULL,
        1'000'000'000ULL,
        10'000'000'000ULL,
        100'000'000'000ULL
    };

    cout << "Threads: " << omp_get_max_threads() << endl;

    ofstream file("monte_carlo_pi_results.csv");
    file << "N,pi_est,abs_error,execution_time_sec,time_per_point\n";

    for (auto N : N_values) {

        cout << "\nComputing N = " << N << endl;

        unsigned long long inside = 0;

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel reduction(+:inside)
        {
            uint64_t seed = 123456789 + omp_get_thread_num();

            #pragma omp for schedule(static)
            for (unsigned long long i = 0; i < N; i += 4) {

                __m256d x, y, r2;

                double x_arr[4], y_arr[4];

                for (int k = 0; k < 4; ++k) {
                    x_arr[k] = fast_double(seed);
                    y_arr[k] = fast_double(seed);
                }

                x = _mm256_loadu_pd(x_arr);
                y = _mm256_loadu_pd(y_arr);

                __m256d x2 = _mm256_mul_pd(x, x);
                __m256d y2 = _mm256_mul_pd(y, y);
                r2 = _mm256_add_pd(x2, y2);

                __m256d one = _mm256_set1_pd(1.0);
                __m256d mask = _mm256_cmp_pd(r2, one, _CMP_LE_OQ);

                int m = _mm256_movemask_pd(mask);

                inside += __builtin_popcount(m);
            }
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        double pi_est = 4.0 * inside / (double)N;
        double error = fabs(pi_est - M_PI);
        double time_per_point = elapsed.count() / (double)N;

        cout << fixed << setprecision(12);
        cout << defaultfloat; //можна закоментувати
        cout << "pi_est = " << pi_est << endl;
        cout << "abs_error = " << error << endl;
        cout << "time = " << elapsed.count() << " sec\n";
        cout << "time_per_point = " << time_per_point << endl;

        file << fixed << setprecision(11)//<<defaultfloat //Можна розкоментувати,щоб зайві нулі прибрати
             << N << ","
             << pi_est << ","
             << error << ","
             << elapsed.count() << ","
             << time_per_point << "\n";
    }

    file.close();
    cout << "\nResults saved.\n";

    return 0;
}
