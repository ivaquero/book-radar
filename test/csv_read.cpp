#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int get_max_row_num(std::ifstream &file) {
  int row_num = 0;
  std::string line;

  while (getline(file, line)) {
    row_num++;
  }

  printf("row_num: %d\n", row_num);
  return row_num;
}

std::vector<std::vector<int>> get_delimeter_indexes(std::ifstream &file,
                                                    std::string line) {
  int loc = 0;
  int start = 0;
  int row_idx = 0;

  std::vector<std::vector<int>> delim_idxes;

  while (getline(file, line)) {
    printf("line: %d\n", row_idx);
    while (line.find(",") != -1) {
      printf("line: %s\n", line.c_str());
      // get the index of the first delimiter
      loc = line.find(",");
      start += loc;
      // printf("start: %d\n", start);
      delim_idxes[row_idx].emplace_back(start);
      // get the rest of the line
      line = line.substr(loc + 1, line.length());
    }
    row_idx++;
  }
  return delim_idxes;
}

void get_value_by_index(std::ifstream &file, int row_idx, int col_idx) {
  std::string line;
  int loc = 0;
  int start = 0;
}

int main() {
  std::ifstream iFile;

  const int MAX_ROW_NUM = 100;
  std::string line;
  std::vector<std::vector<int>> delim_idxes;

  int skip_row = 1;
  int row_idx = 0;
  int col_idx = 0;

  iFile.open("../data/waveband.csv");
  if (!iFile.is_open()) {
    printf("open file failed: %d\n", errno);
    exit(1);
  }

  int row_num = get_max_row_num(iFile);
  delim_idxes = get_delimeter_indexes(iFile, line);

  // 跳过行数
  for (auto i = 0; i < skip_row + 1; i++) {
    getline(iFile, line);
  }

  iFile.close();
  return 0;
}
