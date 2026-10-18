#include <iostream>
#include "redev.h"

const std::string timeout="8";

auto makeRedev(int dim, redev::LOs& ranks, redev::Reals& cuts, bool isRendezvous) {
  redev::Config config;
  config.Global().Set("adios2.bp4", "OpenTimeoutSecs", timeout);
  if(static_cast<redev::ProcessType>(isRendezvous) == redev::ProcessType::Server) {
    auto ptn = redev::RCBPtn(dim,ranks,cuts);
    return redev::Redev(MPI_COMM_WORLD,redev::Partition{ptn},
                        redev::ProcessType::Server, false, std::move(config));
  }
  return redev::Redev(MPI_COMM_WORLD, redev::ProcessType::Client, false,
                      std::move(config));
}

void client() {
  const auto dim = 1;
  auto ranks = redev::LOs(1);
  auto cuts = redev::Reals(1);
  const bool isRendezvous = false;
  auto rdv = makeRedev(dim, ranks, cuts, isRendezvous);
  auto channel = rdv.CreateChannel("foo");
  auto commPair = channel.CreateComm<redev::LO>("foo", MPI_COMM_WORLD);
}

void server() {
  const auto dim = 1;
  auto ranks = redev::LOs({0});
  auto cuts = redev::Reals({0});
  const bool isRendezvous=true;
  auto rdv = makeRedev(dim, ranks, cuts, isRendezvous);
  auto channel = rdv.CreateChannel("foo");
  auto commPair = channel.CreateComm<redev::LO>("foo", MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int nproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REDEV_ALWAYS_ASSERT(nproc == 1);
  REDEV_ALWAYS_ASSERT(argc == 2);
  const int isRdv = atoi(argv[1]);
  std::cout << "isRdv " << isRdv << "\n";
  REDEV_ALWAYS_ASSERT(isRdv == 1 || isRdv == 0);
  if(isRdv)
    server();
  else
    client();
  std::cout << "done\n";
  MPI_Finalize();
  return 0;
}
