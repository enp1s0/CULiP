#include <iostream>
#include <regex>
#include <string>

int main(int argc, char **argv) {
  const std::regex regex(R"(\[CULiP ExpStats\] (.): (.+))");
  std::string buffer;
  while (std::getline(std::cin, buffer)) {
    std::smatch match;
    if (std::regex_match(buffer, match, regex)) {
      const auto input_matrix = match[1];
      const auto result_json = match[2];
      std::cout << input_matrix << ":" << result_json << std::endl;
    }
  }
}
