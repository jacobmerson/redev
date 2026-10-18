#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include "redev.h"
#include "redev_bootstrap.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  if(argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <1=isRendezvousApp,0=isParticipant>\n";
    exit(EXIT_FAILURE);
  }
  int nproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REDEV_ALWAYS_ASSERT(nproc == 1);
  const auto isRdv = atoi(argv[1]);
  {
  const auto dim = 1;
  auto ranks = isRdv ? redev::LOs({0}) : redev::LOs(1);
  auto cuts = isRdv ? redev::Reals({0}) : redev::Reals(1);
  auto ptn = redev::RCBPtn(dim, ranks, cuts);
  // The server decides the transport; the client's SST setting must be
  // ignored, otherwise the two sides would not connect.
  redev::Config config;
  if(isRdv) {
    config.Global().SetTransport(redev::TransportType::BP4);
  } else {
    config.Global().SetTransport(redev::TransportType::SST);
  }
  redev::Redev rdv(MPI_COMM_WORLD, redev::Partition{std::move(ptn)},
                   static_cast<redev::ProcessType>(isRdv), false,
                   std::move(config));
  const std::string name = "foo";
  auto channel = rdv.CreateChannel(name);
  auto commPair = channel.CreateComm<redev::LO>(name, MPI_COMM_WORLD);
  redev::LOs dest{0};
  redev::LOs offsets{0, 1};
  if(!isRdv) {
    commPair.SetOutMessageLayout(dest, offsets);
    redev::LOs msgs(1, 42);
    channel.SendPhase([&]() { commPair.Send(msgs.data()); });
    auto reply = channel.ReceivePhase([&]() { return commPair.Recv(); });
    REDEV_ALWAYS_ASSERT(reply[0] == 1337);
    const auto unused = rdv.GetConfig().Unused();
    REDEV_ALWAYS_ASSERT(std::find(unused.begin(), unused.end(),
                                  "global.transport") != unused.end());
    // the client removes the bootstrap file once it has read it
    REDEV_ALWAYS_ASSERT(!std::filesystem::exists(redev::BootstrapFileName(name)));
  } else {
    auto msgs = channel.ReceivePhase([&]() { return commPair.Recv(); });
    REDEV_ALWAYS_ASSERT(msgs[0] == 42);
    commPair.SetOutMessageLayout(dest, offsets);
    redev::LOs reply(1, 1337);
    channel.SendPhase([&]() { commPair.Send(reply.data()); });
    REDEV_ALWAYS_ASSERT(rdv.GetConfig().Unused().empty());
  }
  }
  MPI_Finalize();
  return 0;
}
