#ifndef REDEV__REDEV_ADIOS_COMM_H
#define REDEV__REDEV_ADIOS_COMM_H
#include "redev_adios_comm_decl.h"

namespace redev {

template <typename T> std::vector<T> AdiosComm<T>::Recv(Mode mode) {
  REDEV_FUNCTION_TIMER;
  //REDEV_ALWAYS_ASSERT(channel.InReceiveCommunicationPhase());
  int rank, commSz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &commSz);
  auto t1 = redev::getTime();

  if(!inMsg.knownSizes) {
    auto rdvRanksVar = io.InquireVariable<redev::GO>(name+"_srcRanks");
    assert(rdvRanksVar);
    auto offsetsVar = io.InquireVariable<redev::GO>(name+"_offsets");
    assert(offsetsVar);

    auto offsetsShape = offsetsVar.Shape();
    assert(offsetsShape.size() == 1);
    const auto offSz = offsetsShape[0];
    inMsg.offset.resize(offSz);
    offsetsVar.SetSelection({{0}, {offSz}});
    eng.Get(offsetsVar, inMsg.offset.data());

    auto rdvRanksShape = rdvRanksVar.Shape();
    assert(rdvRanksShape.size() == 1);
    const auto rsrSz = rdvRanksShape[0];
    inMsg.srcRanks.resize(rsrSz);
    rdvRanksVar.SetSelection({{0},{rsrSz}});
    eng.Get(rdvRanksVar, inMsg.srcRanks.data());

    // TODO: Can remove in synchronous mode?
    eng.PerformGets();
    inMsg.start = static_cast<size_t>(inMsg.offset[rank]);
    inMsg.count = static_cast<size_t>(inMsg.offset[rank+1]-inMsg.start);
    inMsg.knownSizes = true;
  }
  auto t2 = redev::getTime();

  auto msgsVar = io.InquireVariable<T>(name);
  assert(msgsVar);
  std::vector<T> msgs(inMsg.count);
  if(inMsg.count) {
    //only call Get with non-zero sized reads
    msgsVar.SetSelection({{inMsg.start}, {inMsg.count}});
    eng.Get(msgsVar, msgs.data());
  }
  if(mode == Mode::Synchronous) {
    eng.PerformGets();
  }

  //if(mode == Mode::Synchronous) {
  //  eng.EndStep();
  //}
  auto t3 = redev::getTime();
  std::chrono::duration<double> r1 = t2-t1;
  std::chrono::duration<double> r2 = t3-t2;
  if(!rank && verbose) {
    fprintf(stderr, "recv knownSizes %d r1(sec.) r2(sec.) %f %f\n",
            inMsg.knownSizes, r1.count(), r2.count());
  }
  return msgs;
}
template <typename T> void AdiosComm<T>::Send(T *msgs, Mode mode) {
  REDEV_FUNCTION_TIMER;
  //REDEV_ALWAYS_ASSERT(channel.InSendCommunicationPhase());
  int rank, commSz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &commSz);
  GOs degree(recvRanks,0); //TODO ideally, this would not be needed
  for( size_t i=0; i<outMsg.dest.size(); i++) {
    auto destRank = outMsg.dest[i];
    assert(destRank < recvRanks);
    degree[destRank] += outMsg.offsets[i+1] - outMsg.offsets[i];
  }
  GOs rdvRankStart(recvRanks,0);
  auto ret = MPI_Exscan(degree.data(), rdvRankStart.data(), recvRanks,
                        getMpiType(redev::GO()), MPI_SUM, comm);
  assert(ret == MPI_SUCCESS);
  if(!rank) {
    //on rank 0 the result of MPI_Exscan is undefined, set it to zero
    rdvRankStart = GOs(recvRanks,0);
  }

  GOs gDegree(recvRanks,0);
  ret = MPI_Allreduce(degree.data(), gDegree.data(), recvRanks,
                      getMpiType(redev::GO()), MPI_SUM, comm);
  assert(ret == MPI_SUCCESS);
  const size_t gDegreeTot = static_cast<size_t>(std::accumulate(gDegree.begin(), gDegree.end(), redev::GO(0)));

  GOs gStart(recvRanks,0);
  redev::exclusive_scan(gDegree.begin(), gDegree.end(), gStart.begin(), redev::GO(0));

  //The messages array has a different length on each rank ('irregular') so we don't
  //define local size and count here.
  adios2::Dims shape{static_cast<size_t>(gDegreeTot)};
  adios2::Dims start{};
  adios2::Dims count{};
  if(!rdvVar) {
    rdvVar = io.DefineVariable<T>(name, shape, start, count);
  }
  assert(rdvVar);
  const auto srcRanksName = name+"_srcRanks";
  //The source rank offsets array is the same on each process ('regular').
  adios2::Dims srShape{static_cast<size_t>(commSz*recvRanks)};
  adios2::Dims srStart{static_cast<size_t>(recvRanks*rank)};
  adios2::Dims srCount{static_cast<size_t>(recvRanks)};

  //send dest rank offsets array from rank 0
  auto offsets = gStart;
  offsets.push_back(gDegreeTot);
  if(!rank) {
    const auto offsetsName = name+"_offsets";
    const auto oShape = offsets.size();
    const auto oStart = 0;
    const auto oCount = offsets.size();
    if(!offsetsVar) {
      offsetsVar = io.DefineVariable<redev::GO>(offsetsName,{oShape},{oStart},{oCount});
      // if we are in sync mode we will peform all puts at the end of the function, otherwise we need to put this now before
      // offsets data goes out of scope
      eng.Put<redev::GO>(offsetsVar, offsets.data(), (mode==Mode::Deferred)?adios2::Mode::Sync:adios2::Mode::Deferred);
    }
  }

  //send source rank offsets array 'rdvRankStart'
  if(!srcRanksVar) {
    srcRanksVar = io.DefineVariable<redev::GO>(srcRanksName, srShape, srStart, srCount);
    assert(srcRanksVar);
    // if we are in sync mode we will peform all puts at the end of the function, otherwise we need to put this now before
    // ranks data goes out of scope
    eng.Put<redev::GO>(srcRanksVar, rdvRankStart.data(),(mode==Mode::Deferred)?adios2::Mode::Sync:adios2::Mode::Deferred);

  }

  //assume one call to pack from each rank for now
  for( size_t i=0; i<outMsg.dest.size(); i++ ) {
    const auto destRank = outMsg.dest[i];
    const auto lStart = gStart[destRank]+rdvRankStart[destRank];
    const auto lCount = outMsg.offsets[i+1]-outMsg.offsets[i];
    if( lCount > 0 ) {
      start = adios2::Dims{static_cast<size_t>(lStart)};
      count = adios2::Dims{static_cast<size_t>(lCount)};
      rdvVar.SetSelection({start,count});
      eng.Put<T>(rdvVar, &(msgs[outMsg.offsets[i]]));
    }
  }
  if(mode == Mode::Synchronous) {
    eng.PerformPuts();
  }
}
}

#endif // REDEV__REDEV_ADIOS_COMM_H
