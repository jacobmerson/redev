#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <string>         // std::stoi
#include "redev_git_version.h"
#include "redev_comm.h"
#include "redev_adios_channel.h"
#include "redev_strings.h"
//
namespace redev {
namespace {
//Wait for the file to be created by the writer.
//Assuming that if 'Streaming' and 'OpenTimeoutSecs' are set then we are in
//BP4 mode.  SST blocks on Open by default.
void waitForEngineCreation(adios2::IO& io) {
  REDEV_FUNCTION_TIMER;
  auto params = io.Parameters();
  bool isStreaming = params.count("Streaming") &&
      redev::isSameCaseInsensitive(params["Streaming"], "ON");
  bool timeoutSet = params.count("OpenTimeoutSecs") && std::stoi(params["OpenTimeoutSecs"]) > 0;
  bool isSST = redev::isSameCaseInsensitive(io.EngineType(), "SST");
  if( (isStreaming && timeoutSet) || isSST ) return;
  std::cout<<"Waiting for BP4 Engine Creation\n";
  // TODO: REDEV LOGGING...LOG INFO that sleeping for engine creation
  std::this_thread::sleep_for(std::chrono::seconds(2));
}
}

void AdiosChannel::Setup(adios2::IO& s2cIO, adios2::Engine& s2cEngine) {
  REDEV_FUNCTION_TIMER;
  // initialize the partition on the client based on how it's set on the server
  if (process_type_ == ProcessType::Server) {
    std::ignore = SendPartitionTypeToClient(s2cIO, s2cEngine);
  }
  else {
    auto partition_index = SendPartitionTypeToClient(s2cIO, s2cEngine);
    ConstructPartitionFromIndex(partition_index);
  }
  CheckVersion(s2cEngine,s2cIO);
  auto status = s2cEngine.BeginStep();
  REDEV_ALWAYS_ASSERT(status == adios2::StepStatus::OK);
  //rendezvous app rank 0 writes partition info and other apps read
  if(!rank_) {
    if(process_type_==ProcessType::Server) {
      std::visit([&](auto&& partition){partition.Write(s2cEngine, s2cIO);}, partition_);
    }
    else {
      std::visit([&](auto&& partition){partition.Read(s2cEngine, s2cIO);}, partition_);
    }
  }
  s2cEngine.EndStep();
  std::visit([&](auto&& partition){partition.Broadcast(comm_);}, partition_);

}

/*
   * return the number of processes in the client's MPI communicator
 */
redev::LO
AdiosChannel::SendClientCommSizeToServer(adios2::IO& c2sIO, adios2::Engine& c2sEngine) {
  REDEV_FUNCTION_TIMER;
  int commSize;
  MPI_Comm_size(comm_, &commSize);
  const auto varName = "redev client communicator size";
  auto status = c2sEngine.BeginStep();
  REDEV_ALWAYS_ASSERT(status == adios2::StepStatus::OK);
  redev::LO clientCommSz = 0;
  if(process_type_ == ProcessType::Client) {
    auto var = c2sIO.DefineVariable<redev::LO>(varName);
    if(!rank_)
      c2sEngine.Put(var, commSize);
  } else {
    auto var = c2sIO.InquireVariable<redev::LO>(varName);
    if(var && !rank_) {
      c2sEngine.Get(var, clientCommSz);
      c2sEngine.PerformGets(); //default read mode is deferred
    }
  }
  c2sEngine.EndStep();
  if(process_type_==ProcessType::Server)
    redev::Broadcast(&clientCommSz,1,0,comm_);
  return clientCommSz;
}

/*
   * return the number of processes in the server's MPI communicator
 */
redev::LO
AdiosChannel::SendServerCommSizeToClient(adios2::IO& s2cIO, adios2::Engine& s2cEngine) {
  REDEV_FUNCTION_TIMER;
  int commSize;
  MPI_Comm_size(comm_, &commSize);
  const auto varName = "redev server communicator size";
  auto status = s2cEngine.BeginStep();
  REDEV_ALWAYS_ASSERT(status == adios2::StepStatus::OK);
  redev::LO serverCommSz = 0;
  if(process_type_==ProcessType::Server) {
    auto var = s2cIO.DefineVariable<redev::LO>(varName);
    if(!rank_)
      s2cEngine.Put(var, commSize);
  } else {
    auto var = s2cIO.InquireVariable<redev::LO>(varName);
    if(var && !rank_) {
      s2cEngine.Get(var, serverCommSz);
      s2cEngine.PerformGets(); //default read mode is deferred
    }
  }
  s2cEngine.EndStep();
  if(process_type_ == ProcessType::Client)
    redev::Broadcast(&serverCommSz,1,0,comm_);
  return serverCommSz;
}

std::size_t
AdiosChannel::SendPartitionTypeToClient(adios2::IO& s2cIO, adios2::Engine& s2cEngine) {
  REDEV_FUNCTION_TIMER;
  const auto varName = "redev partition type";
  auto status = s2cEngine.BeginStep();
  REDEV_ALWAYS_ASSERT(status == adios2::StepStatus::OK);
  std::size_t partition_index = partition_.index();
  if(process_type_==ProcessType::Server) {
    auto var = s2cIO.DefineVariable<std::size_t>(varName);
    if(!rank_)
      s2cEngine.Put(var, partition_index);
  } else {
    auto var = s2cIO.InquireVariable<std::size_t>(varName);
    if(var && !rank_) {
      s2cEngine.Get(var, partition_index);
      s2cEngine.PerformGets(); //default read mode is deferred
    }
  }
  s2cEngine.EndStep();
  if(process_type_ == ProcessType::Client) {
    redev::Broadcast(&partition_index,1,0,comm_);
  }
  return partition_index;
}

void AdiosChannel::ConstructPartitionFromIndex(size_t partition_index) {
  if(partition_.index() != partition_index) {
    switch(partition_index) {
    case 0:
      partition_.emplace<ClassPtn>();
      REDEV_ALWAYS_ASSERT(partition_.index() == 0ULL);
      break;
    case 1:
      partition_.emplace<RCBPtn>();
      REDEV_ALWAYS_ASSERT(partition_.index() == 1ULL);
      break;
    default:
      Redev_Assert_Fail("Unhandled partition type");
    }
  }
}

void AdiosChannel::CheckVersion(adios2::Engine& eng, adios2::IO& io) {
  REDEV_FUNCTION_TIMER;
  const auto hashVarName = "redev git hash";
  auto status = eng.BeginStep();
  REDEV_ALWAYS_ASSERT(status == adios2::StepStatus::OK);
  //rendezvous app writes the version it has and other apps read
  if(process_type_==ProcessType::Server) {
    auto varVersion = io.DefineVariable<std::string>(hashVarName);
    if(!rank_)
      eng.Put(varVersion, std::string(redevGitHash));
  }
  else {
    auto varVersion = io.InquireVariable<std::string>(hashVarName);
    std::string inHash;
    if(varVersion && !rank_) {
      eng.Get(varVersion, inHash);
      eng.PerformGets(); //default read mode is deferred
      REDEV_ALWAYS_ASSERT(inHash == redevGitHash);
    }
  }
  eng.EndStep();
}

// SST support
// - with a rendezvous + non-rendezvous application pair
// - with only a rendezvous application for debugging/testing
void AdiosChannel::openEnginesSST(bool noClients,
                                  std::string s2cName, std::string c2sName,
                                  adios2::IO& s2cIO, adios2::IO& c2sIO,
                                  adios2::Engine& s2cEngine, adios2::Engine& c2sEngine) {
  REDEV_FUNCTION_TIMER;
  //create one engine's reader and writer pair at a time - SST blocks on open(read)
  if(process_type_==ProcessType::Server) {
    s2cEngine = s2cIO.Open(s2cName, adios2::Mode::Write);
  } else {
    s2cEngine = s2cIO.Open(s2cName, adios2::Mode::Read);
  }
  REDEV_ALWAYS_ASSERT(s2cEngine);
  if(process_type_==ProcessType::Server) {
    if(noClients==false) { //support unit testing
      c2sEngine = c2sIO.Open(c2sName, adios2::Mode::Read);
    }
  } else {
    c2sEngine = c2sIO.Open(c2sName, adios2::Mode::Write);
  }
  REDEV_ALWAYS_ASSERT(c2sEngine);
}

// BP4 support
// - with a rendezvous + non-rendezvous application pair
// - with only a rendezvous application for debugging/testing
// - in streaming and non-streaming modes; non-streaming requires 'waitForEngineCreation'
void AdiosChannel::openEnginesBP4(bool noClients,
                                  std::string s2cName, std::string c2sName,
                                  adios2::IO& s2cIO, adios2::IO& c2sIO,
                                  adios2::Engine& s2cEngine, adios2::Engine& c2sEngine) {
  REDEV_FUNCTION_TIMER;
  //create the engine writers at the same time - BP4 does not wait for the readers (SST does)
  if(process_type_ == ProcessType::Server) {
    s2cEngine = s2cIO.Open(s2cName, adios2::Mode::Write);
    REDEV_ALWAYS_ASSERT(s2cEngine);
  } else {
    c2sEngine = c2sIO.Open(c2sName, adios2::Mode::Write);
    REDEV_ALWAYS_ASSERT(c2sEngine);
  }
  waitForEngineCreation(s2cIO);
  waitForEngineCreation(c2sIO);
  //create engines for reading
  if(process_type_ == ProcessType::Server) {
    if(noClients==false) { //support unit testing
      c2sEngine = c2sIO.Open(c2sName, adios2::Mode::Read);
      REDEV_ALWAYS_ASSERT(c2sEngine);
    }
  } else {
    s2cEngine = s2cIO.Open(s2cName, adios2::Mode::Read);
    REDEV_ALWAYS_ASSERT(s2cEngine);
  }
}
AdiosChannel::AdiosChannel(adios2::ADIOS &adios, MPI_Comm comm,
                           std::string name, adios2::Params params,
                           TransportType transportType, ProcessType processType,
                           Partition &partition, std::string path,
                           bool noClients)
    : comm_(comm), process_type_(processType), partition_(partition)

{
  REDEV_FUNCTION_TIMER;
  MPI_Comm_rank(comm, &rank_);
  auto s2cName = path + name + "_s2c";
  auto c2sName = path + name + "_c2s";
  s2c_io_ = adios.DeclareIO(s2cName);
  c2s_io_ = adios.DeclareIO(c2sName);
  if (transportType == TransportType::SST && noClients == true) {
    // TODO log message here
    transportType = TransportType::BP4;
  }
  std::string engineType;
  switch (transportType) {
  case TransportType::BP4:
    engineType = "BP4";
    s2cName = s2cName + ".bp";
    c2sName = c2sName + ".bp";
    break;
  case TransportType::SST:
    engineType = "SST";
    break;
    // no default case. This will cause a compiler error if we do not handle a
    // an engine type that has been defined in the TransportType enum.
    // (-Werror=switch)
  }
  s2c_io_.SetEngine(engineType);
  c2s_io_.SetEngine(engineType);
  s2c_io_.SetParameters(params);
  c2s_io_.SetParameters(params);
  REDEV_ALWAYS_ASSERT(s2c_io_.EngineType() == c2s_io_.EngineType());
  switch (transportType) {
  case TransportType::SST:
    openEnginesSST(noClients, s2cName, c2sName, s2c_io_, c2s_io_, s2c_engine_,
                   c2s_engine_);
    break;
  case TransportType::BP4:
    openEnginesBP4(noClients, s2cName, c2sName, s2c_io_, c2s_io_, s2c_engine_,
                   c2s_engine_);
    break;
  }
  // TODO pull begin/end step out of Setup/SendReceive metadata functions
  // begin step
  // send metadata
  Setup(s2c_io_, s2c_engine_);
  num_server_ranks_ = SendServerCommSizeToClient(s2c_io_, s2c_engine_);
  num_client_ranks_ = SendClientCommSizeToServer(c2s_io_, c2s_engine_);
  // end step
}
AdiosChannel::~AdiosChannel() {
  REDEV_FUNCTION_TIMER;
  // NEED TO CHECK that the engine exists before trying to close it because it
  // could be in a moved from state
  if (s2c_engine_) {
    s2c_engine_.Close();
  }
  if (c2s_engine_) {
    c2s_engine_.Close();
  }
}
void AdiosChannel::BeginSendCommunicationPhase() {
  REDEV_FUNCTION_TIMER;
  adios2::StepStatus status;
  switch (process_type_) {
  case ProcessType::Client:
    status = c2s_engine_.BeginStep();
    break;
  case ProcessType::Server:
    status = s2c_engine_.BeginStep();
    break;
  }
  REDEV_ALWAYS_ASSERT(status == adios2::StepStatus::OK);
}
void AdiosChannel::EndSendCommunicationPhase() {
  switch (process_type_) {
  case ProcessType::Client:
    c2s_engine_.EndStep();
    break;
  case ProcessType::Server:
    s2c_engine_.EndStep();
    break;
  }
}
void AdiosChannel::BeginReceiveCommunicationPhase() {
  REDEV_FUNCTION_TIMER;
  adios2::StepStatus status;
  switch (process_type_) {
  case ProcessType::Client:
    status = s2c_engine_.BeginStep();
    break;
  case ProcessType::Server:
    status = c2s_engine_.BeginStep();
    break;
  }
  REDEV_ALWAYS_ASSERT(status == adios2::StepStatus::OK);
}
void AdiosChannel::EndReceiveCommunicationPhase() {
  REDEV_FUNCTION_TIMER;
  switch (process_type_) {
  case ProcessType::Client:
    s2c_engine_.EndStep();
    break;
  case ProcessType::Server:
    c2s_engine_.EndStep();
    break;
  }
}

}