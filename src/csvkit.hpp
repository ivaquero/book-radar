#ifndef CSVKIT_H
#define CSVKIT_H

#include <fstream>
#include <string>
#include <vector>

namespace csvkit {

enum struct QuoteState { UnquotedField, QuotedField, QuotedQuote };

namespace parser {

/*!
 * @brief         Parse Quoted Fields
 *
 * @param         row
 * @return        fileds
 * @attention
 */
inline std::vector<std::string> parse_row(const std::string &row) {
  QuoteState state = QuoteState::UnquotedField;
  std::vector<std::string> fields{""};
  size_t idx = 0;

  for (char field_end : row) {
    switch (state) {
    case QuoteState::UnquotedField:
      switch (field_end) {
      case ',':
        fields.push_back("");
        idx++;
        break;
      case '"':
        state = QuoteState::QuotedField;
        break;
      default:
        fields[idx].push_back(field_end);
        break;
      }
      break;
    case QuoteState::QuotedField:
      switch (field_end) {
      case '"':
        state = QuoteState::QuotedQuote;
        break;
      default:
        fields[idx].push_back(field_end);
        break;
      }
      break;
    case QuoteState::QuotedQuote:
      switch (field_end) {
      case ',': // , after closing quote
        fields.push_back("");
        idx++;
        state = QuoteState::UnquotedField;
        break;
      case '"': // "" -> "
        fields[idx].push_back('"');
        state = QuoteState::QuotedField;
        break;
      default: // end of quote
        state = QuoteState::UnquotedField;
        break;
      }
      break;
    }
  }
  return fields;
}
} // namespace parser

namespace reader {
/*!
 * @brief         Read CSV file. Accept "quoted fields ""with quotes"""
 *
 * @param         in
 * @return        table (std::vector<std::vector<std::string>>)
 * @attention
 */
inline std::vector<std::vector<std::string>> read_csv(std::istream &file) {
  std::vector<std::vector<std::string>> table;
  std::string row = "";
  while (!file.eof()) {
    std::getline(file, row);
    if (file.bad() || file.fail()) {
      break;
    }
    auto fields = parser::parse_row(row);
    table.emplace_back(fields);
  }
  return table;
}
} // namespace reader

namespace writer {}

} // namespace csvkit

#endif
