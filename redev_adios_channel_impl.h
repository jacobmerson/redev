#ifndef REDEV__REDEV_ADIOS_CHANNEL_IMPL_H
#define REDEV__REDEV_ADIOS_CHANNEL_IMPL_H
namespace redev {
template <typename T>
BidirectionalComm<T> AdiosChannel::CreateComm(std::string name, MPI_Comm comm) {
  REDEV_FUNCTION_TIMER;
  // TODO, remove s2c/c2s destinction on variable names then use std::move
  // name
  if(comm != MPI_COMM_NULL) {
    auto s2c = std::make_unique<AdiosComm<T>>(comm, num_client_ranks_,
        s2c_engine_, s2c_io_, name);
    auto c2s = std::make_unique<AdiosComm<T>>(comm, num_server_ranks_,
        c2s_engine_, c2s_io_, name);
    switch (process_type_) {
    case ProcessType::Client:
      return {std::move(c2s), std::move(s2c)};
    case ProcessType::Server:
      return {std::move(s2c), std::move(c2s)};
    }
  }
  return {std::make_unique<NoOpComm<T>>(), std::make_unique<NoOpComm<T>>()};
}
}

#endif // REDEV__REDEV_ADIOS_CHANNEL_IMPL_H
