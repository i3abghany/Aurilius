#pragma once
#include <limits>
#include <cmath>
#include <string>
#include <omp.h>
#include "../Aurilius.h"

namespace Aurilius {
    constexpr static double EPS = std::numeric_limits<double>::epsilon() * 1e5;
    namespace {
		template<typename Callable>
		double traprule(const Callable& f, double a, double b, std::size_t order) {
			int i;
			double h = (b - a) / order;
			double y = (f(a) + f(b)) / 2.0;
			double x;
			for (i = 1, x = a + h; i < order; i++, x += h) y += f(x);
			return h * y;
		}
	}

	template<typename Callable>
	double trapezoidal(const Callable & f, double initial_a, double initial_b, std::size_t order = 1000000) {
		int i;
		constexpr std::size_t n_threads = 8;
		double res = 0.0;
		double partial_res, h;
		h = 1.0 / n_threads;
		double a = initial_a, b = initial_b;
		#pragma omp parallel num_threads(n_threads) private(i, a, b, partial_res, res) default(none)
		{
            i = omp_get_thread_num();
			a = initial_a + i * h;
			b = initial_a + ((double)i + 1) * h;
			partial_res = traprule(f, a, b, order);
			#pragma omp critical
			res += partial_res;
		};

		return res;
	}

	template<typename Callable>
	double simpson(const Callable & f, double a, double b, std::size_t order = 100000) {
		const double delta = (b - a) / (1.0 * order);
		double curr_x = a;
		double ans = f(curr_x);
		curr_x += delta;
		bool alt = true;
		for (; std::fabs(curr_x - b) > EPS; curr_x += delta) {
			if (alt) ans += 4 * f(curr_x);
			else ans += 2 * f(curr_x);

			alt = !alt;
		}
		ans += f(curr_x + delta);
		ans *= (delta / 3.0);
		return ans;
	}
}