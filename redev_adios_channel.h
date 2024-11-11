#ifndef REDEV_REDEV_ADIOS_CHANNEL_H
#define REDEV_REDEV_ADIOS_CHANNEL_H
#include "redev_adios_comm.h"
#include "redev_assert.h"
#include "redev_bidirectional_comm.h"
#include "redev_partition.h"
#include "redev_profile.h"
#include <adios2.h>

namespace redev {

class AdiosChannel {
public:
  AdiosChannel(adios2::ADIOS &adios, MPI_Comm comm, std::string name,
               adios2::Params params, TransportType transportType,
               ProcessType processType, Partition &partition, std::string path,
               bool noClients = false);
  // don't allow copying of class because it creates
  AdiosChannel(const AdiosChannel &) = delete;
  AdiosChannel operator=(const AdiosChannel &) = delete;
  // FIXME
  AdiosChannel(AdiosChannel &&o)
      : s2c_io_(std::exchange(o.s2c_io_, adios2::IO())),
        c2s_io_(std::exchange(o.c2s_io_, adios2::IO())),
        c2s_engine_(std::exchange(o.c2s_engine_, adios2::Engine())),
        s2c_engine_(std::exchange(o.s2c_engine_, adios2::Engine())),
        num_client_ranks_(o.num_client_ranks_),
        num_server_ranks_(o.num_server_ranks_),
        comm_(std::exchange(o.comm_, MPI_COMM_NULL)),
        process_type_(o.process_type_), rank_(o.rank_),
        partition_(o.partition_) {REDEV_FUNCTION_TIMER;}
  AdiosChannel operator=(AdiosChannel &&) = delete;
  // FIXME IMPL RULE OF 5
  ~AdiosChannel();
  template <typename T>
  [[nodiscard]] BidirectionalComm<T> CreateComm(std::string name, MPI_Comm comm);

  // TODO s2c/c2s Engine/IO -> send/receive Engine/IO. This removes need for all
  // the switch statements...
  void BeginSendCommunicationPhase();
  void EndSendCommunicationPhase();
  void BeginReceiveCommunicationPhase();
  void EndReceiveCommunicationPhase();

private:
  void openEnginesBP4(bool noClients, std::string s2cName, std::string c2sName,
                      adios2::IO &s2cIO, adios2::IO &c2sIO,
                      adios2::Engine &s2cEngine, adios2::Engine &c2sEngine);
  void openEnginesSST(bool noClients, std::string s2cName, std::string c2sName,
                      adios2::IO &s2cIO, adios2::IO &c2sIO,
                      adios2::Engine &s2cEngine, adios2::Engine &c2sEngine);
  [[nodiscard]] redev::LO SendServerCommSizeToClient(adios2::IO &s2cIO,
                                                     adios2::Engine &s2cEngine);
  [[nodiscard]] redev::LO SendClientCommSizeToServer(adios2::IO &c2sIO,
                                                     adios2::Engine &c2sEngine);
  [[nodiscard]] std::size_t
  SendPartitionTypeToClient(adios2::IO &s2cIO, adios2::Engine &s2cEngine);
  void Setup(adios2::IO &s2cIO, adios2::Engine &s2cEngine);
  void CheckVersion(adios2::Engine &eng, adios2::IO &io);
  void ConstructPartitionFromIndex(size_t partition_index);

  adios2::IO s2c_io_;
  adios2::IO c2s_io_;
  adios2::Engine s2c_engine_;
  adios2::Engine c2s_engine_;
  redev::LO num_client_ranks_;
  redev::LO num_server_ranks_;
  MPI_Comm comm_;
  ProcessType process_type_;
  int rank_;
  Partition &partition_;
};
} // namespace redev

#include "redev_adios_channel_impl.h"

#endif // REDEV__REDEV_ADIOS_CHANNEL_H
