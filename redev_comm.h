#pragma once
#include "redev_assert.h"
#include "redev_exclusive_scan.h"
#include "redev_profile.h"
#include "redev_types.h"
#include <numeric> // accumulate, exclusive_scan
#include <stddef.h>
#include <type_traits> // is_same
#include "redev_time.h"
#include <mpi.h>

namespace redev {

  namespace detail {
    template <typename... T> struct dependent_always_false : std::false_type {};
  }

enum class Mode {
  Deferred,
  Synchronous
};

template<class T>
[[ nodiscard ]]
constexpr MPI_Datatype getMpiType(T) noexcept {
  if constexpr (std::is_same_v<T, char>) { return MPI_CHAR; }
  else if constexpr (std::is_same_v<T, signed short int>) { return MPI_SHORT; }
  else if constexpr (std::is_same_v<T, signed int>) { return MPI_INT; }
  else if constexpr (std::is_same_v<T, signed long>) { return MPI_LONG; }
  else if constexpr (std::is_same_v<T, signed long long>) { return MPI_LONG_LONG; }
  else if constexpr (std::is_same_v<T, signed char>) { return MPI_SIGNED_CHAR; }
  else if constexpr (std::is_same_v<T, unsigned char>) { return MPI_UNSIGNED_CHAR; }
  else if constexpr (std::is_same_v<T, unsigned short>) { return MPI_UNSIGNED_SHORT; }
  else if constexpr (std::is_same_v<T, unsigned int>) { return MPI_UNSIGNED; }
  else if constexpr (std::is_same_v<T, unsigned long>) { return MPI_UNSIGNED_LONG; }
  else if constexpr (std::is_same_v<T, unsigned long long>) { return MPI_UNSIGNED_LONG_LONG; }
  else if constexpr (std::is_same_v<T, float>) { return MPI_FLOAT; }
  else if constexpr (std::is_same_v<T, double>) { return MPI_DOUBLE; }
  else if constexpr (std::is_same_v<T, long double>) { return MPI_LONG_DOUBLE; }
  else if constexpr (std::is_same_v<T, wchar_t>) { return MPI_WCHAR; }
  else if constexpr (std::is_same_v<T, int8_t>) { return MPI_INT8_T; }
  else if constexpr (std::is_same_v<T, int16_t>) { return MPI_INT16_T; }
  else if constexpr (std::is_same_v<T, int32_t>) { return MPI_INT32_T; }
  else if constexpr (std::is_same_v<T, int64_t>) { return MPI_INT64_T; }
  else if constexpr (std::is_same_v<T, uint8_t>) { return MPI_UINT8_T; }
  else if constexpr (std::is_same_v<T, uint16_t>) { return MPI_UINT16_T; }
  else if constexpr (std::is_same_v<T, uint32_t>) { return MPI_UINT32_T; }
  else if constexpr (std::is_same_v<T, uint64_t>) { return MPI_UINT64_T; }
  else if constexpr (std::is_same_v<T, bool>) { return MPI_CXX_BOOL; }
  else if constexpr (std::is_same_v<T, std::complex<float>>) { return MPI_CXX_FLOAT_COMPLEX; }
  else if constexpr (std::is_same_v<T, std::complex<double>>) { return MPI_CXX_DOUBLE_COMPLEX; }
  else if constexpr (std::is_same_v<T, std::complex<long double>>) { return MPI_CXX_LONG_DOUBLE_COMPLEX; }
  else{ static_assert(detail::dependent_always_false<T>::value, "type has unkown map to MPI_Type"); return {}; }
  // empty return statement needed to avoid compiler warning
  return {};
}

template<typename T>
void Broadcast(T* data, int count, int root, MPI_Comm comm) {
  REDEV_FUNCTION_TIMER;
  auto type = getMpiType(T());
  MPI_Bcast(data, count, type, root, comm);
}

/**
 * The InMessageLayout struct contains the arrays defining the arrangement of
 * data in the array returned by Communicator::Recv.
 */
struct InMessageLayout {
  /**
   * Array of source ranks sized NumberOfClientRanks*NumberOfServerRanks.  Each
   * rank reads the entire array once at the start of a communication round.
   * A communication round is defined as a series of sends and receives using
   * the same message layout.
   */
  redev::GOs srcRanks;
  /**
   * Array of size NumberOfReceiverRanks+1 that indicates the segment of the
   * messages array each server rank should read. NumberOfReceiverRanks is
   * defined as the number of ranks calling Communicator::Recv.
   */
  redev::GOs offset;
  /**
   * Set to true if Communicator::Recv has been called and the message layout data set;
   * false otherwise.
   */
  bool knownSizes;
  /**
   * Index into the messages array (returned by Communicator::Recv) where the current process should start
   * reading.
   */
  size_t start;
  /**
   * Number of items (of the user specified type passed to the template
   * parameter of AdiosComm) that should be read from the messages array
   * (returned by Communicator::Recv).
   */
  size_t count;
};

/**
 * The Communicator class provides an abstract interface for sending and
 * receiving messages to/from the client and server.
 * TODO: Split Communicator into Send/Recieve Communicators, bidirectional constructed by composition and can perform both send and receive
 */
template<typename T>
class Communicator {
  public:
    /**
     * Set the arrangement of data in the messages array so that its segments,
     * defined by the offsets array, are sent to the correct destination ranks,
     * defined by the dest array.
     * @param[in] dest array of integers specifying the destination rank for a
     * portion of the msgs array
     * @param[in] offsets array of length |dest|+1 defining the segment of the
     * msgs array (passed to the Send function) being sent to each destination rank.
     * the segment [ msgs[offsets[i]] : msgs[offsets[i+1]] } is sent to rank dest[i]
     */
    virtual void SetOutMessageLayout(LOs& dest, LOs& offsets) = 0;
    /**
     * Send the array.
     * @param[in] msgs array of data to be sent according to the layout specified
     *            with SetOutMessageLayout
     */
    virtual void Send(T *msgs, Mode mode) = 0;
    /**
     * Receive an array. Use AdiosComm's GetInMessageLayout to retreive
     * an instance of the InMessageLayout struct containing the layout of
     * the received array.
     */
    virtual std::vector<T> Recv(Mode mode) = 0;

    virtual InMessageLayout GetInMessageLayout() = 0;
    virtual ~Communicator() = default;
};

template <typename T>
class NoOpComm : public Communicator<T> {
    void SetOutMessageLayout(LOs& dest, LOs& offsets) final {};
    void Send(T *msgs, Mode /*unused*/) final {};
    std::vector<T> Recv(Mode /*unused*/) final { return {}; }
    InMessageLayout GetInMessageLayout() final { return {}; }
};

}
