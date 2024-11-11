#include <redev.h>
#include <cassert>
#include "redev_assert.h"
#include "redev_git_version.h"
#include "redev.h"
#include "redev_profile.h"
#include "redev_exclusive_scan.h"
#include <algorithm>     // find_if

namespace redev {

  //TODO consider moving the ClassPtn source to another file
  ClassPtn::ClassPtn() {
    REDEV_FUNCTION_TIMER;
  }

  ClassPtn::ClassPtn(MPI_Comm comm, const redev::LOs& ranks_, const ModelEntVec& ents) {
    REDEV_FUNCTION_TIMER;
    REDEV_ALWAYS_ASSERT(comm != MPI_COMM_NULL);
    assert(ranks_.size() == ents.size());
    if( ! ModelEntDimsValid(ents) ) exit(EXIT_FAILURE);
    for(auto i=0; i<ranks_.size(); i++) {
      modelEntToRank[ents[i]] = ranks_[i];
    }
    Gather(comm);
  }

  void ClassPtn::Gather(MPI_Comm comm, int root) {
    REDEV_FUNCTION_TIMER;
    int rank, commSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &commSize);
    auto degree = !rank ? redev::LOs(commSize+1) : redev::LOs();
    auto serialized = SerializeModelEntsAndRanks();
    int len = static_cast<int>(serialized.size());
    MPI_Gather(&len,1,MPI_INT,degree.data(),1,MPI_INT,root,comm);
    if(root==rank) {
      auto offset = redev::LOs(commSize+1);
      redev::exclusive_scan(degree.begin(), degree.end(), offset.begin(), redev::LO(0));
      auto allSerialized = redev::LOs(offset.back());
      MPI_Gatherv(serialized.data(), len, MPI_INT, allSerialized.data(),
          degree.data(), offset.data(), MPI_INT, root, comm);
      modelEntToRank = DeserializeModelEntsAndRanks(allSerialized);
    } else {
      MPI_Gatherv(serialized.data(), len, MPI_INT, NULL, NULL, NULL, MPI_INT, root, comm);
    }
  }

  bool ClassPtn::ModelEntDimsValid(const ClassPtn::ModelEntVec& ents) const {
    REDEV_FUNCTION_TIMER;
    auto badDim = [](ModelEnt e){ return (e.first<0 || e.first>3); };
    auto res = std::find_if(begin(ents), end(ents), badDim);
    if (res != std::end(ents)) {
      std::stringstream ss;
      ss << "ERROR: a ModelEnt contains an invalid dimension: "
         << res->first << "... exiting\n";
      std::cerr << ss.str();
    }
    return (res == std::end(ents));
  }

  redev::LO ClassPtn::GetRank(ModelEnt ent) const {
    REDEV_FUNCTION_TIMER;
    REDEV_ALWAYS_ASSERT(ent.first>=0 && ent.first <=3); //check for valid dimension
    assert(modelEntToRank.size());
    assert(modelEntToRank.count(ent));
    return modelEntToRank.at(ent);
  }

  redev::LOs ClassPtn::GetRanks() const {
    REDEV_FUNCTION_TIMER;
    redev::LOs ranks(modelEntToRank.size());
    int i=0;
    for(const auto& iter : modelEntToRank) {
      ranks[i++]=iter.second;
    }
    return ranks;
  }

  ClassPtn::ModelEntVec ClassPtn::GetModelEnts() const {
    REDEV_FUNCTION_TIMER;
    ModelEntVec ents(modelEntToRank.size());
    int i=0;
    for(const auto& iter : modelEntToRank) {
      ents[i++]=iter.first;
    }
    return ents;
  }

  redev::LOs ClassPtn::SerializeModelEntsAndRanks() const {
    REDEV_FUNCTION_TIMER;
    const auto stride = 3;
    const auto numEnts = modelEntToRank.size();
    redev::LOs entsAndRanks;
    entsAndRanks.reserve(numEnts*stride);
    for(const auto& iter : modelEntToRank) {
      auto ent = iter.first;
      entsAndRanks.push_back(ent.first);   //dim
      entsAndRanks.push_back(ent.second);  //id
      entsAndRanks.push_back(iter.second); //rank
    }
    REDEV_ALWAYS_ASSERT(entsAndRanks.size()==numEnts*stride);
    return entsAndRanks;
  }

  ClassPtn::ModelEntToRank ClassPtn::DeserializeModelEntsAndRanks(const redev::LOs& serialized) const {
    REDEV_FUNCTION_TIMER;
    const auto stride = 3;
    REDEV_ALWAYS_ASSERT(serialized.size()%stride==0);
    ModelEntToRank me2r;
    for(size_t i=0; i<serialized.size(); i+=stride) {
      const auto dim=serialized[i];
      REDEV_ALWAYS_ASSERT(dim>=0 && dim<=3);
      const auto id=serialized[i+1];
      const auto rank=serialized[i+2];
      ModelEnt ent(dim,id);
      const auto hasEnt = me2r.count(ent);
      if( !hasEnt ) {
        me2r[ent] = rank;
      } else {
        REDEV_ALWAYS_ASSERT(rank == me2r[ent]);
      }
    }
    return me2r;
  }

  void ClassPtn::Write(adios2::Engine& eng, adios2::IO& io) {
    REDEV_FUNCTION_TIMER;
    auto serialized = SerializeModelEntsAndRanks();
    const auto len = serialized.size();
    auto entsAndRanksVar = io.DefineVariable<redev::LO>(entsAndRanksVarName,{},{},{len});
    eng.Put(entsAndRanksVar, serialized.data());
    eng.PerformPuts();
  }

  void ClassPtn::Read(adios2::Engine& eng, adios2::IO& io) {
    REDEV_FUNCTION_TIMER;
    const auto step = eng.CurrentStep();
    auto entsAndRanksVar = io.InquireVariable<redev::LO>(entsAndRanksVarName);
    assert(entsAndRanksVar);

    auto blocksInfo = eng.BlocksInfo(entsAndRanksVar,step);
    assert(blocksInfo.size()==1);
    entsAndRanksVar.SetBlockSelection(blocksInfo[0].BlockID);
    redev::LOs serialized;
    eng.Get(entsAndRanksVar, serialized);
    eng.PerformGets(); //default read mode is deferred

    modelEntToRank = DeserializeModelEntsAndRanks(serialized);
  }

  void ClassPtn::Broadcast(MPI_Comm comm, int root) {
    REDEV_FUNCTION_TIMER;
    int rank;
    MPI_Comm_rank(comm, &rank);
    auto serialized = SerializeModelEntsAndRanks();
    int len = static_cast<int>(serialized.size());
    redev::Broadcast(&len, 1, root, comm);
    if(root != rank) {
      serialized.resize(len);
    }
    redev::Broadcast(serialized.data(), serialized.size(), root, comm);
    if(root != rank) {
      modelEntToRank = DeserializeModelEntsAndRanks(serialized);
    }
  }


  //TODO consider moving the RCBPtn source to another file
  RCBPtn::RCBPtn() {
    REDEV_FUNCTION_TIMER;
  }

  RCBPtn::RCBPtn(redev::LO dim_)
    : dim(dim_) {
    REDEV_FUNCTION_TIMER;
    assert(dim>0 && dim<=3);
  }

  RCBPtn::RCBPtn(redev::LO dim_, std::vector<int>& ranks_, std::vector<double>& cuts_)
    : dim(dim_), ranks(ranks_), cuts(cuts_) {
    REDEV_FUNCTION_TIMER;
    assert(dim>0 && dim<=3);
  }

  redev::LO RCBPtn::GetRank(std::array<redev::Real,3>& pt) const { //TODO better name?
    REDEV_FUNCTION_TIMER;
    assert(ranks.size() && cuts.size());
    assert(dim>0 && dim<=3);
    const auto len = cuts.size();
    const auto levels = std::log2(len);
    auto lvl = 0;
    auto idx = 1;
    auto d = 0;
    while(lvl < levels) {
      if(pt[d]<cuts.at(idx))
        idx = idx*2;
      else
        idx = idx*2+1;
      ++lvl;
      d = (d + 1) % dim;
    }
    auto rankIdx = idx-std::pow(2,lvl);
    assert(rankIdx < ranks.size());
    return ranks.at(rankIdx);
  }

  std::vector<redev::LO> RCBPtn::GetRanks() const {
    REDEV_FUNCTION_TIMER;
    return ranks;
  }

  std::vector<redev::Real> RCBPtn::GetCuts() const {
    REDEV_FUNCTION_TIMER;
    return cuts;
  }

  void RCBPtn::Write(adios2::Engine& eng, adios2::IO& io) {
    REDEV_FUNCTION_TIMER;
    const auto len = ranks.size();
    if(!len) return; //don't attempt zero length write
    assert(len==cuts.size());
    auto ranksVar = io.DefineVariable<redev::LO>(ranksVarName,{},{},{len});
    auto cutsVar = io.DefineVariable<redev::Real>(cutsVarName,{},{},{len});
    eng.Put(ranksVar, ranks.data());
    eng.Put(cutsVar, cuts.data());
    eng.PerformPuts();
  }

  void RCBPtn::Read(adios2::Engine& eng, adios2::IO& io) {
    REDEV_FUNCTION_TIMER;
    const auto step = eng.CurrentStep();
    auto ranksVar = io.InquireVariable<redev::LO>(ranksVarName);
    auto cutsVar = io.InquireVariable<redev::Real>(cutsVarName);
    assert(ranksVar && cutsVar);

    auto blocksInfo = eng.BlocksInfo(ranksVar,step);
    assert(blocksInfo.size()==1);
    ranksVar.SetBlockSelection(blocksInfo[0].BlockID);
    eng.Get(ranksVar, ranks);

    blocksInfo = eng.BlocksInfo(ranksVar,step);
    assert(blocksInfo.size()==1);
    ranksVar.SetBlockSelection(blocksInfo[0].BlockID);
    eng.Get(cutsVar, cuts);
    eng.PerformGets(); //default read mode is deferred
  }

  void RCBPtn::Broadcast(MPI_Comm comm, int root) {
    REDEV_FUNCTION_TIMER;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int count = ranks.size();
    redev::Broadcast(&count, 1, root, comm);
    if(root != rank) {
      ranks.resize(count);
      cuts.resize(count);
    }
    redev::Broadcast(ranks.data(), ranks.size(), root, comm);
    redev::Broadcast(cuts.data(), cuts.size(), root, comm);
  }

  Redev::Redev(MPI_Comm comm, Partition ptn, ProcessType processType, bool noClients)
    : comm(comm), adios(comm), ptn(ptn), processType(processType), noClients(noClients) {
    PERFSTUBS_INITIALIZE();
    REDEV_FUNCTION_TIMER;
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REDEV_ALWAYS_ASSERT(isInitialized);
    UpdateRank();
  }
  Redev::Redev(MPI_Comm comm, ProcessType processType, bool noClients)
    : comm(comm), adios(comm), ptn(), processType(processType), noClients(noClients) {
      PERFSTUBS_INITIALIZE();
      REDEV_FUNCTION_TIMER;
      REDEV_ALWAYS_ASSERT(processType == ProcessType::Client);
      int isInitialized = 0;
      MPI_Initialized(&isInitialized);
      REDEV_ALWAYS_ASSERT(isInitialized);
      UpdateRank();
    }

  ProcessType Redev::GetProcessType() const noexcept { return processType; }
  const Partition &Redev::GetPartition() const noexcept {return ptn;}
  bool Redev::RankParticipates() const noexcept { return comm != MPI_COMM_NULL; }
  void Redev::UpdateRank() {
    if(RankParticipates()) {
      MPI_Comm_rank(comm, &rank); //set member var
    }
    else {
      rank = -1;
    }
  }
  MPI_Comm Redev::GetMPIComm() const noexcept { return comm; }
  Channel Redev::CreateAdiosChannel(std::string name, adios2::Params params,
                                    TransportType transportType,
                                    std::string path) {
    REDEV_FUNCTION_TIMER;
    if(RankParticipates()) {
      return AdiosChannel{
          adios,       comm, std::move(name), std::move(params), transportType,
          processType, ptn,  std::move(path), noClients};
    }
    return NoOpChannel{};
  }

  }
