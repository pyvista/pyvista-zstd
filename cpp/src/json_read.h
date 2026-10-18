// A reader for the two JSON metadata documents this format carries -- members of
// a compact json.dumps, not a general JSON parser. In a header because the write,
// append and read paths must agree on what a member of those documents is.

#ifndef PVZSTD_JSON_READ_H
#define PVZSTD_JSON_READ_H

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace pvzstd::json {

inline bool SkipWs(const std::string &s, size_t *i) {
  while (*i < s.size() && (s[*i] == ' ' || s[*i] == '\n' || s[*i] == '\t' || s[*i] == '\r')) ++*i;
  return *i < s.size();
}

// Reads the four hex digits at s[at]; false if any is missing or not hex.
inline bool ReadHex4(const std::string &s, size_t at, unsigned *code) {
  if (at + 4 > s.size()) return false;
  *code = 0;
  for (size_t k = 0; k < 4; ++k) {
    const char h = s[at + k];
    *code <<= 4;
    if (h >= '0' && h <= '9')
      *code |= static_cast<unsigned>(h - '0');
    else if (h >= 'a' && h <= 'f')
      *code |= static_cast<unsigned>(h - 'a' + 10);
    else if (h >= 'A' && h <= 'F')
      *code |= static_cast<unsigned>(h - 'A' + 10);
    else
      return false;
  }
  return true;
}

// Appends a code point to *out as UTF-8.
inline void AppendUtf8(std::string *out, unsigned code) {
  if (code < 0x80) {
    out->push_back(static_cast<char>(code));
  } else if (code < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (code >> 6)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (code >> 12)));
    out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (code >> 18)));
    out->push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
  }
}

// Reads a string starting at the opening quote; leaves *i past the close.
inline bool ReadJsonString(const std::string &s, size_t *i, std::string *out) {
  if (*i >= s.size() || s[*i] != '"') return false;
  ++*i;
  out->clear();
  while (*i < s.size()) {
    const char c = s[*i];
    if (c == '"') {
      ++*i;
      return true;
    }
    if (c == '\\') {
      if (*i + 1 >= s.size()) return false;
      const char esc = s[*i + 1];
      switch (esc) {
        case '"':
          out->push_back('"');
          break;
        case '\\':
          out->push_back('\\');
          break;
        case '/':
          out->push_back('/');
          break;
        case 'n':
          out->push_back('\n');
          break;
        case 'r':
          out->push_back('\r');
          break;
        case 't':
          out->push_back('\t');
          break;
        case 'b':
          out->push_back('\b');
          break;
        case 'f':
          out->push_back('\f');
          break;
        case 'u': {
          unsigned code = 0;
          if (!ReadHex4(s, *i + 2, &code)) return false;
          *i += 4;
          // json.dumps writes a code point above U+FFFF as a surrogate pair; a lone
          // surrogate is kept as its own three-byte sequence, as surrogatepass would.
          if (code >= 0xD800 && code <= 0xDBFF && *i + 7 < s.size() && s[*i + 2] == '\\' &&
              s[*i + 3] == 'u') {
            unsigned low = 0;
            if (ReadHex4(s, *i + 4, &low) && low >= 0xDC00 && low <= 0xDFFF) {
              code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
              *i += 6;
            }
          }
          AppendUtf8(out, code);
          break;
        }
        default:
          return false;
      }
      *i += 2;
      continue;
    }
    out->push_back(c);
    ++*i;
  }
  return false;
}

// Advances *i past the value beginning at *i, whatever kind it is.
inline bool SkipValue(const std::string &s, size_t *i) {
  if (!SkipWs(s, i)) return false;
  const char c = s[*i];
  if (c == '"') {
    std::string ignored;
    return ReadJsonString(s, i, &ignored);
  }
  if (c == '{' || c == '[') {
    const char open = c;
    const char close = (c == '{') ? '}' : ']';
    int depth = 0;
    while (*i < s.size()) {
      const char d = s[*i];
      if (d == '"') {
        std::string ignored;
        if (!ReadJsonString(s, i, &ignored)) return false;
        continue;
      }
      if (d == open)
        ++depth;
      else if (d == close && --depth == 0) {
        ++*i;
        return true;
      }
      ++*i;
    }
    return false;
  }
  // number, true, false, null
  while (*i < s.size() && s[*i] != ',' && s[*i] != '}' && s[*i] != ']') ++*i;
  return true;
}

// Position of a top-level member's value, or npos. Scans members rather than
// searching for `"key":`, which a user-supplied array name could contain.
inline size_t FindMember(const std::string &s, const char *key) {
  size_t i = 0;
  if (!SkipWs(s, &i) || s[i] != '{') return std::string::npos;
  ++i;
  while (SkipWs(s, &i)) {
    if (s[i] == '}') return std::string::npos;
    if (s[i] == ',') {
      ++i;
      continue;
    }
    std::string name;
    if (!ReadJsonString(s, &i, &name)) return std::string::npos;
    if (!SkipWs(s, &i) || s[i] != ':') return std::string::npos;
    ++i;
    if (!SkipWs(s, &i)) return std::string::npos;
    if (name == key) return i;
    if (!SkipValue(s, &i)) return std::string::npos;
  }
  return std::string::npos;
}

inline bool MemberString(const std::string &s, const char *key, std::string *out) {
  size_t i = FindMember(s, key);
  if (i == std::string::npos) return false;
  return ReadJsonString(s, &i, out);
}

inline bool MemberInt(const std::string &s, const char *key, long long *out) {
  size_t i = FindMember(s, key);
  if (i == std::string::npos) return false;
  const size_t start = i;
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
  size_t digits = 0;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
    ++i;
    ++digits;
  }
  if (digits == 0) return false;
  *out = std::strtoll(s.substr(start, i - start).c_str(), nullptr, 10);
  return true;
}

inline bool MemberStringArray(const std::string &s, const char *key,
                              std::vector<std::string> *out) {
  size_t i = FindMember(s, key);
  if (i == std::string::npos || i >= s.size() || s[i] != '[') return false;
  ++i;
  out->clear();
  while (SkipWs(s, &i)) {
    if (s[i] == ']') return true;
    if (s[i] == ',') {
      ++i;
      continue;
    }
    std::string item;
    if (!ReadJsonString(s, &i, &item)) return false;
    out->push_back(std::move(item));
  }
  return false;
}

// The member's opening brace and the index just past its closing brace.
inline bool MemberObjectSpan(const std::string &s, const char *key, size_t *open,
                             size_t *past_close) {
  size_t i = FindMember(s, key);
  if (i == std::string::npos || i >= s.size() || s[i] != '{') return false;
  *open = i;
  if (!SkipValue(s, &i)) return false;
  *past_close = i;
  return true;
}

// Keys of the object opening at `open`, in document order -- the order the
// reference reader reports, building a dict from the same document.
inline bool ObjectKeys(const std::string &s, size_t open, std::vector<std::string> *out) {
  size_t i = open;
  if (i >= s.size() || s[i] != '{') return false;
  ++i;
  out->clear();
  while (SkipWs(s, &i)) {
    if (s[i] == '}') return true;
    if (s[i] == ',') {
      ++i;
      continue;
    }
    std::string name;
    if (!ReadJsonString(s, &i, &name)) return false;
    if (!SkipWs(s, &i) || s[i] != ':') return false;
    ++i;
    if (!SkipValue(s, &i)) return false;
    out->push_back(std::move(name));
  }
  return false;
}

}  // namespace pvzstd::json

#endif  // PVZSTD_JSON_READ_H
