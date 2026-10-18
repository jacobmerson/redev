#include "redev_bootstrap.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace redev {

namespace {

constexpr const char* kHeader = "redev-bootstrap";

[[noreturn]] void Fail(const std::string& msg)
{
  throw std::runtime_error("redev bootstrap: " + msg);
}

double ParseSeconds(const ResolvedSettings& settings, const std::string& key)
{
  const auto& value = settings.at(key).value;
  try {
    std::size_t used = 0;
    const double seconds = std::stod(value, &used);
    if (used == value.size() && seconds > 0) {
      return seconds;
    }
  } catch (const std::exception&) {
  }
  Fail("redev." + key + " must be a positive number of seconds, not '" +
       value + "'");
}

BootstrapData Parse(const std::string& contents, const std::string& file)
{
  std::istringstream in(contents);
  std::string line;
  std::string header;
  int version = 0;
  if (!std::getline(in, line) || !(std::istringstream(line) >> header >> version) ||
      header != kHeader) {
    Fail(file + " is not a redev bootstrap file");
  }
  if (version > kBootstrapVersion) {
    Fail(file + " has version " + std::to_string(version) +
         "; this redev reads up to version " +
         std::to_string(kBootstrapVersion));
  }
  BootstrapData data;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const auto eq = line.find('=');
    if (eq == std::string::npos || eq == 0) {
      Fail(file + " has a malformed line '" + line + "'");
    }
    data[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return data;
}

void BroadcastString(std::string& s, MPI_Comm comm)
{
  unsigned long long size = s.size();
  MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
  if (size > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    Fail("bootstrap data is too large");
  }
  s.resize(size);
  MPI_Bcast(s.data(), static_cast<int>(size), MPI_CHAR, 0, comm);
}

} // namespace

std::string BootstrapFileName(const std::string& section)
{
  return section + ".redev-bootstrap";
}

BootstrapSettings GetBootstrapSettings(const Config& config,
                                       const std::string& section)
{
  const auto settings = config.Get(section, "redev");
  BootstrapSettings result;
  result.timeoutSecs = ParseSeconds(settings, "BootstrapTimeoutSecs");
  result.pollSecs = ParseSeconds(settings, "BootstrapPollSecs");
  return result;
}

void SendBootstrap(const std::string& section, const BootstrapData& data,
                   MPI_Comm comm)
{
  for (const auto& [key, value] : data) {
    if (key.empty() || key.find_first_of("=\n") != std::string::npos ||
        value.find('\n') != std::string::npos) {
      Fail("invalid bootstrap entry '" + key + "'");
    }
  }
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  if (rank) {
    return;
  }
  const std::filesystem::path file = BootstrapFileName(section);
  if (file.has_parent_path()) {
    std::filesystem::create_directories(file.parent_path());
  }
  const auto tmp = file.string() + ".tmp." + std::to_string(getpid());
  {
    std::ofstream out(tmp);
    out << kHeader << " " << kBootstrapVersion << "\n";
    for (const auto& [key, value] : data) {
      out << key << "=" << value << "\n";
    }
    if (!out) {
      Fail("cannot write " + tmp);
    }
  }
  std::error_code ec;
  std::filesystem::rename(tmp, file, ec);
  if (ec) {
    std::remove(tmp.c_str());
    Fail("cannot create " + file.string() + ": " + ec.message());
  }
}

BootstrapData ReceiveBootstrap(const std::string& section,
                               const BootstrapSettings& settings,
                               MPI_Comm comm)
{
  const auto file = BootstrapFileName(section);
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int found = 0;
  std::string contents;
  if (!rank) {
    using Clock = std::chrono::steady_clock;
    const auto deadline =
      Clock::now() + std::chrono::duration<double>(settings.timeoutSecs);
    while (true) {
      std::ifstream in(file);
      if (in) {
        contents.assign(std::istreambuf_iterator<char>(in),
                        std::istreambuf_iterator<char>());
        found = 1;
        break;
      }
      if (Clock::now() >= deadline) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::duration<double>(settings.pollSecs));
    }
    if (found) {
      std::remove(file.c_str());
    }
  }
  MPI_Bcast(&found, 1, MPI_INT, 0, comm);
  if (!found) {
    Fail("timed out after " + std::to_string(settings.timeoutSecs) +
         " s waiting for " + file);
  }
  BroadcastString(contents, comm);
  return Parse(contents, file);
}

} // namespace redev
