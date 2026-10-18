#include "redev_assert.h"
#include "redev_bootstrap.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

template <typename F>
bool throws(F&& f)
{
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

const redev::BootstrapSettings fast{0.5, 0.01};

void writeRaw(const std::string& section, const std::string& contents,
              MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (!rank) {
    std::ofstream(redev::BootstrapFileName(section)) << contents;
  }
  MPI_Barrier(comm);
}

void testRoundTrip(MPI_Comm comm)
{
  const std::string section = "bootstrap_dir/app";
  redev::SendBootstrap(section, {{"transport", "bp4"}, {"extra", "a=b c"}},
                          comm);
  MPI_Barrier(comm);
  const auto data = redev::ReceiveBootstrap(section, fast, comm);
  REDEV_ALWAYS_ASSERT(data.size() == 2);
  REDEV_ALWAYS_ASSERT(data.at("transport") == "bp4");
  REDEV_ALWAYS_ASSERT(data.at("extra") == "a=b c");
  // the receiver removes the file
  REDEV_ALWAYS_ASSERT(
    !std::filesystem::exists(redev::BootstrapFileName(section)));
  MPI_Barrier(comm);
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (!rank) {
    std::filesystem::remove_all("bootstrap_dir");
  }
}

void testErrors(MPI_Comm comm)
{
  // nothing published
  REDEV_ALWAYS_ASSERT(
    throws([&] { std::ignore = redev::ReceiveBootstrap("missing", fast, comm); }));
  // a newer format version
  writeRaw("newer", "redev-bootstrap 2\ntransport=bp4\n", comm);
  REDEV_ALWAYS_ASSERT(
    throws([&] { std::ignore = redev::ReceiveBootstrap("newer", fast, comm); }));
  // not a bootstrap file
  writeRaw("garbage", "hello\n", comm);
  REDEV_ALWAYS_ASSERT(
    throws([&] { std::ignore = redev::ReceiveBootstrap("garbage", fast, comm); }));
  // a line without '='
  writeRaw("malformed", "redev-bootstrap 1\ntransport\n", comm);
  REDEV_ALWAYS_ASSERT(
    throws([&] { std::ignore = redev::ReceiveBootstrap("malformed", fast, comm); }));
  // invalid entries are rejected before anything is written
  REDEV_ALWAYS_ASSERT(throws([&] {
    redev::SendBootstrap("bad", {{"a=b", "1"}}, comm);
  }));
  REDEV_ALWAYS_ASSERT(throws([&] {
    redev::SendBootstrap("bad", {{"a", "1\n2"}}, comm);
  }));
  REDEV_ALWAYS_ASSERT(!std::filesystem::exists(redev::BootstrapFileName("bad")));
}

void testSettings()
{
  redev::Config config;
  auto s = redev::GetBootstrapSettings(config, "a");
  REDEV_ALWAYS_ASSERT(s.timeoutSecs == 60);
  REDEV_ALWAYS_ASSERT(s.pollSecs == 0.05);
  redev::Config custom;
  custom.Channel("b").Set("redev", "BootstrapTimeoutSecs", "5");
  REDEV_ALWAYS_ASSERT(redev::GetBootstrapSettings(custom, "b").timeoutSecs == 5);
  redev::Config bad;
  bad.Global().Set("redev", "BootstrapPollSecs", "soon");
  REDEV_ALWAYS_ASSERT(
    throws([&] { std::ignore = redev::GetBootstrapSettings(bad, "c"); }));
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  testRoundTrip(MPI_COMM_WORLD);
  testErrors(MPI_COMM_WORLD);
  testSettings();
  MPI_Finalize();
  return 0;
}
