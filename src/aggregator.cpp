#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <utility>
#include <algorithm>
#include "params.hpp"

struct aggregation_t {
	std::string entry_name;
	std::size_t time_sum;
	std::size_t time_max;
	std::size_t time_min;
	std::size_t count;

	void add(
		const std::size_t time
		) {
		time_sum += time;
		time_max = std::max(time, time_max);
		time_min = std::min(time, time_min);
		count++;
	}

	aggregation_t(
		const std::string entry_name
		) : entry_name(entry_name), time_sum(0), time_max(0), time_min(UINT64_MAX), count(0) {}
};

void add_to_aggregation_list(
	std::vector<aggregation_t>& aggregation_list,
	const std::string& entry_name,
	const std::size_t elapsed_time
	) {
	// If the function has already pushed to aggregation list, just add elapsed time to it
	for (auto& aggregation : aggregation_list) {
		if (aggregation.entry_name == entry_name) {
			aggregation.add(elapsed_time);
			return;
		}
	}

	// If the function name is not found, push it to the list
	aggregation_t new_entry(entry_name);
	new_entry.add(elapsed_time);
	aggregation_list.push_back(new_entry);
}

void add_to_detail_aggregation_list(
	std::vector<std::tuple<std::string, std::vector<aggregation_t>>>& detail_aggregation_list,
	const std::string& category_name,
	const std::string& entry_name,
	const std::size_t elapsed_time
	) {
	for (auto& aggregation : detail_aggregation_list) {
		if (std::get<0>(aggregation) == category_name) {
			add_to_aggregation_list(std::get<1>(aggregation), entry_name, elapsed_time);
			return;
		}
	}

	std::vector<aggregation_t> new_list;
	add_to_aggregation_list(new_list, entry_name, elapsed_time);

	detail_aggregation_list.push_back(
		std::make_tuple(category_name, new_list)
		);
}

std::vector<aggregation_t> get_detail_aggregation(
	std::vector<std::tuple<std::string, std::vector<aggregation_t>>>& detail_aggregation_list,
	const std::string& category_name
	) {
	for (auto& aggregation : detail_aggregation_list) {
		if (std::get<0>(aggregation) == category_name) {
			return std::get<1>(aggregation);
		}
	}
	return std::vector<aggregation_t>{};
}

int main() {
	std::printf(
		"#####################################\n"
		"#       CULiP Profiling Result      #\n"
		"#  https://github.com/enp1s0/CULiP  #\n"
		"#####################################\n"
		);
	std::string line;

	// Time aggregation per function catehory
	std::vector<aggregation_t> category_aggregation_list;

	// Time aggregation detail
	std::vector<std::tuple<std::string /*category name*/, std::vector<aggregation_t>>> detail_aggregation_list;

	// For find CULiP result lines
	std::regex CULiP_result_regex(std::string(R"(\[)") + CULIP_RESULT_PREFIX + R"(\]\[([a-zA-Z0-9_]+)-(.*)\] ([0-9]+)ns$)");
	std::smatch CULiP_smatch;

	// string length
	std::size_t max_category_name_length = 0;
	std::size_t max_detail_name_length = 10;

	// time sum
	std::size_t time_sum = 0;

	// line number
	std::size_t line_number = 1;

	while (std::getline(std::cin, line)) {
		if (std::regex_search(line, CULiP_smatch, CULiP_result_regex)) {
			// CULiP_smatch[1] : Function name
			// CULiP_smatch[2] : Function params
			// CULiP_smatch[3] : time [ns]

			if (CULiP_smatch.size() < 4) {
				std::fprintf(stderr, "[Error] \"%s\" (l.%lu) is not valid result\n", line.c_str(), line_number);
				return 1;
			}

			const auto function_name = CULiP_smatch[1];
			const auto function_params = CULiP_smatch[2];
			const auto elapsed_time = std::stoull(CULiP_smatch[3]);
			
			// Add to list
			add_to_aggregation_list(category_aggregation_list, function_name, elapsed_time);
			add_to_detail_aggregation_list(detail_aggregation_list, function_name, function_params, elapsed_time);

			// Update max name length
			max_category_name_length = std::max<std::size_t>(max_category_name_length, function_name.length());
			max_detail_name_length = std::max<std::size_t>(max_detail_name_length, function_params.length());

			// Update time sum
			time_sum += elapsed_time;
		}
	}

	// Sort by category elapsed time
	const auto compare_aggregation_t = [](const aggregation_t& a, const aggregation_t& b){return a.time_sum > b.time_sum;};
	std::sort(category_aggregation_list.begin(), category_aggregation_list.end(), compare_aggregation_t);

	for (const auto& category_aggregation : category_aggregation_list) {
		const auto category_name = category_aggregation.entry_name;
		const auto category_time = category_aggregation.time_sum;

		// category time
		std::printf("\n");
		std::printf("- %*s : [%lu ns; %e s;%6.2f%%]\n", static_cast<int>(max_category_name_length), category_name.c_str(), category_time, category_time * 1e-9, static_cast<double>(category_time) / time_sum * 100);

		auto detail_aggregation = get_detail_aggregation(detail_aggregation_list, category_name);
		std::sort(detail_aggregation.begin(), detail_aggregation.end(), compare_aggregation_t);

		std::printf("  %*s %8s %21s %12s %12s %12s\n", static_cast<int>(max_detail_name_length), "params", "count", "sum", "avg", "max", "min");

		for (const auto& da : detail_aggregation) {

			// category time
			std::printf("  %*s %8lu %10.3fms(%6.2f%%) %10.3fms %10.3fms %10.3fms\n",
						static_cast<int>(max_detail_name_length),
						da.entry_name.c_str(),
						da.count,
						da.time_sum * 1e-6,
						static_cast<double>(da.time_sum) / category_time * 100,
						da.time_sum * 1e-6 / da.count,
						da.time_max * 1e-6,
						da.time_min * 1e-6);
		}
	}
}
