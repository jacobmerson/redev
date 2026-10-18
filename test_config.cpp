#include "redev_assert.h"
#include "redev_config.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
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

bool contains(const std::vector<std::string>& v, const std::string& s)
{
  return std::find(v.begin(), v.end(), s) != v.end();
}

void testBuiltin()
{
  redev::Config c;
  REDEV_ALWAYS_ASSERT(c.GetTransport("a") == redev::TransportType::BP4);
  auto s = c.GetSettings("a", redev::TransportType::BP4);
  REDEV_ALWAYS_ASSERT(s.at("Streaming").value == "On");
  REDEV_ALWAYS_ASSERT(s.at("OpenTimeoutSecs").value == "60");
  REDEV_ALWAYS_ASSERT(s.at("BeginStepPollingFrequencySecs").value == "0.05");
  REDEV_ALWAYS_ASSERT(s.at("Streaming").scope == redev::ConfigScope::Builtin);
  REDEV_ALWAYS_ASSERT(s.at("Streaming").section == "global");
  REDEV_ALWAYS_ASSERT(c.Unused().empty());
}

void testScopeFirst()
{
  using redev::ConfigScope;
  redev::Config c;
  // a higher scope in the global section beats a channel Default
  c.Global().Set("g", "k", "global-file", ConfigScope::File);
  c.Channel("a").Set("g", "k", "channel-default", ConfigScope::Default);
  // within the same scope the channel section wins
  c.Global().Set("g", "j", "global-file", ConfigScope::File);
  c.Channel("a").Set("g", "j", "channel-file", ConfigScope::File);
  // Override always wins
  c.Global().Set("g", "o", "global-override", ConfigScope::Override);
  c.Channel("a").Set("g", "o", "channel-env", ConfigScope::Environment);
  auto s = c.Get("a", "g");
  REDEV_ALWAYS_ASSERT(s.at("k").value == "global-file");
  REDEV_ALWAYS_ASSERT(s.at("k").scope == ConfigScope::File);
  REDEV_ALWAYS_ASSERT(s.at("k").section == "global");
  REDEV_ALWAYS_ASSERT(s.at("j").value == "channel-file");
  REDEV_ALWAYS_ASSERT(s.at("j").section == "a");
  REDEV_ALWAYS_ASSERT(s.at("o").value == "global-override");
  // other channels only see the global section
  REDEV_ALWAYS_ASSERT(c.Get("b", "g").at("j").value == "global-file");
}

void testTransport()
{
  using redev::ConfigScope;
  redev::Config c;
  c.Channel("a").SetTransport(redev::TransportType::SST, ConfigScope::Default);
  REDEV_ALWAYS_ASSERT(c.GetTransport("a") == redev::TransportType::SST);
  REDEV_ALWAYS_ASSERT(c.GetTransport("b") == redev::TransportType::BP4);
  REDEV_ALWAYS_ASSERT(redev::ParseTransport("Bp4") == redev::TransportType::BP4);
  REDEV_ALWAYS_ASSERT(redev::ParseTransport("SST") == redev::TransportType::SST);
  REDEV_ALWAYS_ASSERT(throws([] { std::ignore = redev::ParseTransport("mpi"); }));
}

void testFixedOnceRead()
{
  redev::Config c;
  c.Channel("a").Set("g", "k", "1");
  std::ignore = c.Get("a", "g");
  REDEV_ALWAYS_ASSERT(throws([&] { c.Channel("a").Set("g", "k", "2"); }));
  // global settings read through a channel are fixed too
  REDEV_ALWAYS_ASSERT(throws([&] { c.Global().Set("g", "k", "2"); }));
  // other groups and channels can still change
  c.Channel("a").Set("h", "k", "2");
  c.Channel("b").Set("g", "k", "2");
  std::ignore = c.GetTransport("a");
  REDEV_ALWAYS_ASSERT(throws(
    [&] { c.Channel("a").SetTransport(redev::TransportType::SST); }));
  REDEV_ALWAYS_ASSERT(
    throws([&] { c.Global().SetTransport(redev::TransportType::SST); }));
}

void testUnused()
{
  redev::Config c;
  c.Global().Set("adios2.sst", "DataTransport", "rdma");
  c.Channel("a").Set("adios2.bp4", "OpenTimeoutSecs", "5");
  c.Channel("misspelled").Set("adios2.bp4", "OpenTimeoutSecs", "5");
  c.Channel("misspelled").SetTransport(redev::TransportType::SST);
  std::ignore = c.GetSettings("a", redev::TransportType::BP4);
  auto unused = c.Unused();
  REDEV_ALWAYS_ASSERT(unused.size() == 3);
  REDEV_ALWAYS_ASSERT(contains(unused, "global.adios2.sst.DataTransport"));
  REDEV_ALWAYS_ASSERT(contains(unused, "misspelled.adios2.bp4.OpenTimeoutSecs"));
  REDEV_ALWAYS_ASSERT(contains(unused, "misspelled.transport"));
  std::ostringstream os;
  c.Print(os);
  REDEV_ALWAYS_ASSERT(os.str().find("[a]") != std::string::npos);
  REDEV_ALWAYS_ASSERT(os.str().find("adios2.bp4.OpenTimeoutSecs = 5 (Override, a)") !=
                      std::string::npos);
}

#ifdef REDEV_HAS_YAML
void testYaml(MPI_Comm comm)
{
  using redev::ConfigScope;
  redev::Config c;
  c.LoadString(R"(
global:
  transport: SST
  adios2.sst:
    DataTransport: rdma
channels:
  core/core:
    transport: bp4
    adios2.bp4:
      BeginStepPollingFrequencySecs: 0.1
)",
               ConfigScope::File);
  REDEV_ALWAYS_ASSERT(c.GetTransport("edge/edge") == redev::TransportType::SST);
  REDEV_ALWAYS_ASSERT(c.GetTransport("core/core") == redev::TransportType::BP4);
  auto s = c.GetSettings("core/core", redev::TransportType::BP4);
  REDEV_ALWAYS_ASSERT(s.at("BeginStepPollingFrequencySecs").value == "0.1");
  REDEV_ALWAYS_ASSERT(s.at("BeginStepPollingFrequencySecs").scope ==
                      ConfigScope::File);

  auto bad = [](const std::string& yaml) {
    return throws([&] { redev::Config().LoadString(yaml, ConfigScope::File); });
  };
  REDEV_ALWAYS_ASSERT(bad("global: [1, 2]"));
  REDEV_ALWAYS_ASSERT(bad("global:\n  transport: mpi"));
  REDEV_ALWAYS_ASSERT(bad("global:\n  adios2.bp4:\n    Streaming: [On]"));
  REDEV_ALWAYS_ASSERT(bad("other: 1"));
  REDEV_ALWAYS_ASSERT(bad("global: {"));

  int rank;
  MPI_Comm_rank(comm, &rank);
  const std::string path = "test_config_tmp.yaml";
  if (!rank) {
    std::ofstream f(path);
    f << "channels:\n  a:\n    adios2.bp4:\n      OpenTimeoutSecs: 7\n";
  }
  MPI_Barrier(comm);
  redev::Config fromFile;
  fromFile.Channel("a").Set("adios2.bp4", "OpenTimeoutSecs", "3",
                            ConfigScope::Default);
  fromFile.LoadFile(path, comm);
  REDEV_ALWAYS_ASSERT(fromFile.GetSettings("a", redev::TransportType::BP4)
                        .at("OpenTimeoutSecs")
                        .value == "7");
  MPI_Barrier(comm);
  if (!rank) {
    std::remove(path.c_str());
  }
  REDEV_ALWAYS_ASSERT(
    throws([&] { redev::Config().LoadFile("does_not_exist.yaml", comm); }));

  setenv("REDEV_CONFIG_YAML", "global:\n  adios2.bp4:\n    OpenTimeoutSecs: 9\n",
         1);
  redev::Config fromEnv;
  fromEnv.LoadEnvironment(comm);
  auto e = fromEnv.GetSettings("a", redev::TransportType::BP4).at("OpenTimeoutSecs");
  REDEV_ALWAYS_ASSERT(e.value == "9");
  REDEV_ALWAYS_ASSERT(e.scope == ConfigScope::Environment);
  unsetenv("REDEV_CONFIG_YAML");
}
#else
void testYaml(MPI_Comm comm)
{
  REDEV_ALWAYS_ASSERT(throws(
    [] { redev::Config().LoadString("global: {}", redev::ConfigScope::File); }));
  REDEV_ALWAYS_ASSERT(throws([&] { redev::Config().LoadFile("x.yaml", comm); }));
  setenv("REDEV_CONFIG_YAML", "global: {}", 1);
  REDEV_ALWAYS_ASSERT(throws([&] { redev::Config().LoadEnvironment(comm); }));
  unsetenv("REDEV_CONFIG_YAML");
}
#endif

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  testBuiltin();
  testScopeFirst();
  testTransport();
  testFixedOnceRead();
  testUnused();
  testYaml(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
