#include "redev_config.h"
#include "redev_strings.h"
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#ifdef REDEV_HAS_YAML
#include <yaml-cpp/yaml.h>
#endif

namespace redev {

namespace {

constexpr const char* kGlobal = "global";

[[noreturn]] void Fail(const std::string& msg)
{
  throw std::runtime_error("redev::Config: " + msg);
}

std::size_t Index(ConfigScope scope) { return static_cast<std::size_t>(scope); }

const char* ScopeName(ConfigScope scope)
{
  switch (scope) {
    case ConfigScope::Builtin: return "Builtin";
    case ConfigScope::Default: return "Default";
    case ConfigScope::File: return "File";
    case ConfigScope::Environment: return "Environment";
    case ConfigScope::Override: return "Override";
  }
  return "";
}

std::string SectionLabel(const std::optional<std::string>& channel)
{
  return channel ? *channel : kGlobal;
}

#ifdef REDEV_HAS_YAML
std::string Scalar(const YAML::Node& node, const std::string& where)
{
  if (!node.IsScalar()) {
    Fail(where + " must be a scalar value");
  }
  return node.as<std::string>();
}
#endif

} // namespace

std::string BackendName(TransportType transport)
{
  switch (transport) {
    case TransportType::BP4: return "adios2.bp4";
    case TransportType::SST: return "adios2.sst";
  }
  return "";
}

std::string TransportName(TransportType transport)
{
  switch (transport) {
    case TransportType::BP4: return "bp4";
    case TransportType::SST: return "sst";
  }
  return "";
}

TransportType ParseTransport(const std::string& name)
{
  if (isSameCaseInsensitive(name, "bp4")) {
    return TransportType::BP4;
  }
  if (isSameCaseInsensitive(name, "sst")) {
    return TransportType::SST;
  }
  Fail("unknown transport '" + name + "'");
}

Config::Config()
{
  auto builtin = ConfigScope::Builtin;
  SetTransport(std::nullopt, TransportType::BP4, builtin);
  const auto bp4 = BackendName(TransportType::BP4);
  SetValue(std::nullopt, bp4, "Streaming", "On", builtin);
  SetValue(std::nullopt, bp4, "OpenTimeoutSecs", "60", builtin);
  SetValue(std::nullopt, bp4, "BeginStepPollingFrequencySecs", "0.05",
           builtin);
  SetValue(std::nullopt, BackendName(TransportType::SST), "OpenTimeoutSecs",
           "60", builtin);
  SetValue(std::nullopt, "redev", "BootstrapTimeoutSecs", "60", builtin);
  SetValue(std::nullopt, "redev", "BootstrapPollSecs", "0.05", builtin);
}

Config::Section Config::Global() { return Section(*this, std::nullopt); }

Config::Section Config::Channel(const std::string& name)
{
  return Section(*this, name);
}

void Config::SetTransport(const std::optional<std::string>& channel,
                          TransportType transport, ConfigScope scope)
{
  if (channel) {
    if (readTransports_.count(*channel)) {
      Fail("cannot change the transport of '" + *channel +
           "' after it was read");
    }
    channels_[*channel].transport[Index(scope)] = transport;
  } else {
    if (globalTransportRead_) {
      Fail("cannot change the global transport after it was read");
    }
    global_.transport[Index(scope)] = transport;
  }
}

void Config::SetValue(const std::optional<std::string>& channel,
                      const std::string& group, const std::string& key,
                      std::string value, ConfigScope scope)
{
  const auto label = SectionLabel(channel) + "." + group + "." + key;
  if (channel) {
    if (readGroups_.count({*channel, group})) {
      Fail("cannot change " + label + " after it was read");
    }
    channels_[*channel].groups[group][key][Index(scope)] = std::move(value);
  } else {
    if (readGlobalGroups_.count(group)) {
      Fail("cannot change " + label + " after it was read");
    }
    global_.groups[group][key][Index(scope)] = std::move(value);
  }
}

ResolvedSettings Config::Resolve(const std::string& channel,
                                 const std::string& group) const
{
  const Group* globalGroup = nullptr;
  const Group* channelGroup = nullptr;
  if (auto it = global_.groups.find(group); it != global_.groups.end()) {
    globalGroup = &it->second;
  }
  if (auto ch = channels_.find(channel); ch != channels_.end()) {
    if (auto it = ch->second.groups.find(group); it != ch->second.groups.end()) {
      channelGroup = &it->second;
    }
  }
  std::set<std::string> keys;
  for (const auto* g : {globalGroup, channelGroup}) {
    if (g) {
      for (const auto& [key, values] : *g) {
        keys.insert(key);
      }
    }
  }
  ResolvedSettings resolved;
  for (const auto& key : keys) {
    for (auto i = kNumScopes; i-- > 0;) {
      if (channelGroup) {
        if (auto it = channelGroup->find(key);
            it != channelGroup->end() && it->second[i]) {
          resolved[key] = {*it->second[i], static_cast<ConfigScope>(i), channel};
          break;
        }
      }
      if (globalGroup) {
        if (auto it = globalGroup->find(key);
            it != globalGroup->end() && it->second[i]) {
          resolved[key] = {*it->second[i], static_cast<ConfigScope>(i), kGlobal};
          break;
        }
      }
    }
  }
  return resolved;
}

std::pair<TransportType, ResolvedSetting> Config::ResolveTransport(
  const std::string& channel) const
{
  const auto ch = channels_.find(channel);
  for (auto i = kNumScopes; i-- > 0;) {
    if (ch != channels_.end() && ch->second.transport[i]) {
      auto t = *ch->second.transport[i];
      return {t, {TransportName(t), static_cast<ConfigScope>(i), channel}};
    }
    if (global_.transport[i]) {
      auto t = *global_.transport[i];
      return {t, {TransportName(t), static_cast<ConfigScope>(i), kGlobal}};
    }
  }
  Fail("no transport configured");
}

TransportType Config::GetTransport(const std::string& name) const
{
  globalTransportRead_ = true;
  readTransports_.insert(name);
  return ResolveTransport(name).first;
}

ResolvedSettings Config::Get(const std::string& section,
                             const std::string& group) const
{
  readGlobalGroups_.insert(group);
  readGroups_.insert({section, group});
  return Resolve(section, group);
}

ResolvedSettings Config::GetSettings(const std::string& name,
                                     TransportType transport) const
{
  return Get(name, BackendName(transport));
}

void Config::Print(std::ostream& os) const
{
  auto printSection = [&](const std::string& channel) {
    os << "[" << channel << "]\n";
    const auto [transport, origin] = ResolveTransport(channel);
    os << "  transport = " << origin.value << " (" << ScopeName(origin.scope)
       << ", " << origin.section << ")\n";
    std::set<std::string> groups;
    for (const auto& [group, values] : global_.groups) {
      groups.insert(group);
    }
    if (auto ch = channels_.find(channel); ch != channels_.end()) {
      for (const auto& [group, values] : ch->second.groups) {
        groups.insert(group);
      }
    }
    for (const auto& group : groups) {
      for (const auto& [key, setting] : Resolve(channel, group)) {
        os << "  " << group << "." << key << " = " << setting.value << " ("
           << ScopeName(setting.scope) << ", " << setting.section << ")\n";
      }
    }
  };
  printSection(kGlobal);
  std::set<std::string> channels;
  for (const auto& [channel, data] : channels_) {
    channels.insert(channel);
  }
  for (const auto& [channel, group] : readGroups_) {
    channels.insert(channel);
  }
  for (const auto& channel : channels) {
    printSection(channel);
  }
}

std::vector<std::string> Config::Unused() const
{
  std::vector<std::string> unused;
  auto isSet = [](const auto& perScope) {
    for (auto i = Index(ConfigScope::Default); i < kNumScopes; ++i) {
      if (perScope[i]) {
        return true;
      }
    }
    return false;
  };
  auto collect = [&](const std::string& label, const SectionData& data,
                     bool transportRead, auto&& groupRead) {
    if (isSet(data.transport) && !transportRead) {
      unused.push_back(label + ".transport");
    }
    for (const auto& [group, keys] : data.groups) {
      if (groupRead(group)) {
        continue;
      }
      for (const auto& [key, values] : keys) {
        if (isSet(values)) {
          unused.push_back(label + "." + group + "." + key);
        }
      }
    }
  };
  collect(kGlobal, global_, globalTransportRead_, [&](const std::string& g) {
    return readGlobalGroups_.count(g) > 0;
  });
  for (const auto& [channel, data] : channels_) {
    collect(channel, data, readTransports_.count(channel) > 0,
            [&, &ch = channel](const std::string& g) {
              return readGroups_.count({ch, g}) > 0;
            });
  }
  return unused;
}

void Config::LoadString(const std::string& yaml, ConfigScope scope)
{
#ifdef REDEV_HAS_YAML
  YAML::Node root;
  try {
    root = YAML::Load(yaml);
  } catch (const YAML::Exception& e) {
    Fail(std::string("YAML error: ") + e.what());
  }
  if (!root || root.IsNull()) {
    return;
  }
  if (!root.IsMap()) {
    Fail("the top level must be a mapping");
  }
  auto loadSection = [&](const std::optional<std::string>& channel,
                         const YAML::Node& node) {
    const auto label = SectionLabel(channel);
    if (node.IsNull()) {
      return;
    }
    if (!node.IsMap()) {
      Fail("section " + label + " must be a mapping");
    }
    for (const auto& entry : node) {
      const auto name = Scalar(entry.first, "a key in " + label);
      if (name == "transport") {
        SetTransport(channel,
                     ParseTransport(Scalar(entry.second, label + ".transport")),
                     scope);
        continue;
      }
      const auto& group = entry.second;
      if (group.IsNull()) {
        continue;
      }
      if (!group.IsMap()) {
        Fail(label + "." + name + " must be a mapping");
      }
      for (const auto& setting : group) {
        const auto key = Scalar(setting.first, "a key in " + label + "." + name);
        SetValue(channel, name, key,
                 Scalar(setting.second, label + "." + name + "." + key), scope);
      }
    }
  };
  for (const auto& entry : root) {
    const auto name = Scalar(entry.first, "a top-level key");
    if (name == kGlobal) {
      loadSection(std::nullopt, entry.second);
    } else if (name == "channels") {
      if (entry.second.IsNull()) {
        continue;
      }
      if (!entry.second.IsMap()) {
        Fail("channels must be a mapping");
      }
      for (const auto& channel : entry.second) {
        loadSection(Scalar(channel.first, "a channel name"), channel.second);
      }
    } else {
      Fail("unknown top-level key '" + name + "'");
    }
  }
#else
  (void)yaml;
  (void)scope;
  Fail("YAML configuration requires redev built with REDEV_ENABLE_YAML");
#endif
}

void Config::LoadFile(const std::string& path, MPI_Comm comm)
{
#ifndef REDEV_HAS_YAML
  (void)comm;
  Fail("cannot load " + path +
       ": YAML configuration requires redev built with REDEV_ENABLE_YAML");
#else
  auto read = [&](std::string& contents) {
    std::ifstream file(path);
    if (!file) {
      return false;
    }
    contents.assign(std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>());
    return true;
  };
  std::string contents;
  if (comm == MPI_COMM_NULL) {
    if (!read(contents)) {
      Fail("cannot read configuration file " + path);
    }
    LoadString(contents, ConfigScope::File);
    return;
  }
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int ok = 1;
  if (rank == 0) {
    ok = read(contents) ? 1 : 0;
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
  if (!ok) {
    Fail("cannot read configuration file " + path);
  }
  unsigned long long size = contents.size();
  MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
  if (size > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    Fail("configuration file " + path + " is too large");
  }
  contents.resize(size);
  MPI_Bcast(contents.data(), static_cast<int>(size), MPI_CHAR, 0, comm);
  LoadString(contents, ConfigScope::File);
#endif
}

void Config::LoadEnvironment(MPI_Comm comm)
{
  if (const char* path = std::getenv("REDEV_CONFIG"); path && *path) {
    LoadFile(path, comm);
  }
  if (const char* yaml = std::getenv("REDEV_CONFIG_YAML"); yaml && *yaml) {
    LoadString(yaml, ConfigScope::Environment);
  }
}

Config::Section::Section(Config& config, std::optional<std::string> channel)
  : config_(&config), channel_(std::move(channel))
{
}

Config::Section& Config::Section::SetTransport(TransportType transport,
                                               ConfigScope scope)
{
  config_->SetTransport(channel_, transport, scope);
  return *this;
}

Config::Section& Config::Section::Set(const std::string& group,
                                      const std::string& key, std::string value,
                                      ConfigScope scope)
{
  config_->SetValue(channel_, group, key, std::move(value), scope);
  return *this;
}

Config::Section& Config::Section::Set(
  const std::string& group, const std::map<std::string, std::string>& settings,
  ConfigScope scope)
{
  for (const auto& [key, value] : settings) {
    config_->SetValue(channel_, group, key, value, scope);
  }
  return *this;
}

} // namespace redev
