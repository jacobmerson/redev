#ifndef REDEV__REDEV_ADIOS_COMM_DECL_H
#define REDEV__REDEV_ADIOS_COMM_DECL_H
#include "redev_comm.h"
#include <adios2.h>

namespace {
void checkStep(adios2::StepStatus status) {
  REDEV_ALWAYS_ASSERT(status == adios2::StepStatus::OK);
}
}

namespace redev {

/**
 * The AdiosComm class implements the Communicator interface to support sending
 * messages between the clients and server via ADIOS2.  The BP4 and SST ADIOS2
 * engines are currently supported.
 * One AdiosComm object is required for each communication link direction.  For
 * example, for a client and server to both send and receive messages one
 * AdiosComm for client->server messaging and another AdiosComm for
 * server->client messaging are needed. Redev::BidirectionalComm is a helper
 * class for this use case.
 */
template<typename T>
class AdiosComm : public Communicator<T> {
public:
  /**
     * Create an AdiosComm object.  Collective across sender and receiver ranks.
     * Calls to the constructor from the sender and receiver ranks must be in
     * the same order (i.e., first creating the client-to-server object then the
     * server-to-client link).
     * @param[in] comm_ MPI communicator for sender ranks
     * @param[in] recvRanks_ number of ranks in the receivers MPI communicator
     * @param[in] eng_ ADIOS2 engine for writing on the sender side
     * @param[in] io_ ADIOS2 IO associated with eng_
     * @param[in] name_ unique name among AdiosComm objects
   */
  AdiosComm(MPI_Comm comm_, int recvRanks_, adios2::Engine& eng_, adios2::IO& io_, std::string name_)
      : comm(comm_), recvRanks(recvRanks_), eng(eng_), io(io_), name(name_), verbose(0) {
    inMsg.knownSizes = false;
  }

  /// We are explicitly not allowing copy/move constructor/assignment as we don't
  /// know if the ADIOS2 Engine and IO objects can be safely copied/moved.
  AdiosComm(const AdiosComm& other) = delete;
  AdiosComm(AdiosComm&& other) = delete;
  AdiosComm& operator=(const AdiosComm& other) = delete;
  AdiosComm& operator=(AdiosComm&& other) = delete;

  void SetOutMessageLayout(LOs& dest_, LOs& offsets_) {
    REDEV_FUNCTION_TIMER;
    outMsg = OutMessageLayout{dest_, offsets_};
  }
  void Send(T *msgs, Mode mode);
  std::vector<T> Recv(Mode mode);
  /**
     * Return the InMessageLayout object.
     * @todo should return const object
   */
  InMessageLayout GetInMessageLayout() {
    return inMsg;
  }
  /**
     * Control the amount of output from AdiosComm functions.  The higher the value the more output is written.
     * @param[in] lvl valid values are [0:5] where 0 is silent and 5 is produces
     *                the most output
   */
  void SetVerbose(int lvl) {
    assert(lvl>=0 && lvl<=5);
    verbose = lvl;
  }
private:
  MPI_Comm comm;
  int recvRanks;
  adios2::Engine& eng;
  adios2::IO& io;
  adios2::Variable<T> rdvVar;
  adios2::Variable<redev::GO> srcRanksVar;
  adios2::Variable<redev::GO> offsetsVar;
  std::string name;
  //support only one call to pack for now...
  struct OutMessageLayout {
    LOs dest;
    LOs offsets;
  } outMsg;
  int verbose;
  //receive side state
  InMessageLayout inMsg;
};
}

#endif // REDEV__REDEV_ADIOS_COMM_DECL_H
