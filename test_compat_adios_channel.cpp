#include <cstdlib>
#include <iostream>
#include "redev.h"

// Exercises the deprecated CreateAdiosChannel API, alone and mixed with
// CreateChannel across the server and client.
enum class Mode { BothOld = 0, OldServer = 1, OldClient = 2 };

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  if(argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <1=isRendezvousApp,0=isParticipant> <mode=0|1|2>\n";
    exit(EXIT_FAILURE);
  }
  int nproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REDEV_ALWAYS_ASSERT(nproc == 1);
  const auto isRdv = atoi(argv[1]);
  const auto mode = static_cast<Mode>(atoi(argv[2]));
  const bool useOld = mode == Mode::BothOld ||
                      (mode == Mode::OldServer && isRdv) ||
                      (mode == Mode::OldClient && !isRdv);
  {
  const auto dim = 1;
  auto ranks = isRdv ? redev::LOs({0}) : redev::LOs(1);
  auto cuts = isRdv ? redev::Reals({0}) : redev::Reals(1);
  auto ptn = redev::RCBPtn(dim, ranks, cuts);
  redev::Config config;
  config.Global().Set("adios2.bp4", "OpenTimeoutSecs", "4");
  redev::Redev rdv(MPI_COMM_WORLD, redev::Partition{std::move(ptn)},
                   static_cast<redev::ProcessType>(isRdv), false,
                   std::move(config));
  const std::string name = "foo";
  auto channel = useOld
    ? rdv.CreateAdiosChannel(name, {{"Streaming", "On"}, {"OpenTimeoutSecs", "5"}},
                             redev::TransportType::BP4)
    : rdv.CreateChannel(name);
  if(useOld && isRdv) {
    // params apply at the Override scope; builtin defaults still apply
    auto s = rdv.GetConfig().GetSettings(name, redev::TransportType::BP4);
    REDEV_ALWAYS_ASSERT(s.at("OpenTimeoutSecs").value == "5");
    REDEV_ALWAYS_ASSERT(s.at("OpenTimeoutSecs").scope == redev::ConfigScope::Override);
    REDEV_ALWAYS_ASSERT(s.at("BeginStepPollingFrequencySecs").value == "0.05");
  }
  auto commPair = channel.CreateComm<redev::LO>(name, MPI_COMM_WORLD);
  redev::LOs dest{0};
  redev::LOs offsets{0, 1};
  if(!isRdv) {
    commPair.SetOutMessageLayout(dest, offsets);
    redev::LOs msgs(1, 42);
    channel.SendPhase([&]() { commPair.Send(msgs.data()); });
    auto reply = channel.ReceivePhase([&]() { return commPair.Recv(); });
    REDEV_ALWAYS_ASSERT(reply[0] == 1337);
  } else {
    auto msgs = channel.ReceivePhase([&]() { return commPair.Recv(); });
    REDEV_ALWAYS_ASSERT(msgs[0] == 42);
    commPair.SetOutMessageLayout(dest, offsets);
    redev::LOs reply(1, 1337);
    channel.SendPhase([&]() { commPair.Send(reply.data()); });
  }
  }
  MPI_Finalize();
  return 0;
}
