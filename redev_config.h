#ifndef REDEV_REDEV_CONFIG_H
#define REDEV_REDEV_CONFIG_H
#include "redev_types.h"
#include <array>
#include <map>
#include <mpi.h>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace redev {

/**
 * Where a setting came from. For a given key the highest scope wins,
 * independent of the order in which settings were applied:
 *
 *   Builtin < Default < File < Environment < Override
 *
 * Within the same scope, a channel section wins over the global section.
 *
 * - Builtin: redev's defaults; for "adios2.bp4": Streaming=On,
 *   OpenTimeoutSecs=60, BeginStepPollingFrequencySecs=0.05; for "adios2.sst":
 *   OpenTimeoutSecs=60; for "redev": BootstrapTimeoutSecs=60,
 *   BootstrapPollSecs=0.05
 * - Default: set in code; may be changed by a file or the environment
 * - File: a YAML file (Config::LoadFile, or REDEV_CONFIG)
 * - Environment: YAML text in REDEV_CONFIG_YAML, for quick per-job tuning
 * - Override: set in code; always wins
 */
enum class ConfigScope { Builtin, Default, File, Environment, Override };

/** One setting after merging, with its origin. */
struct ResolvedSetting {
  std::string value;
  ConfigScope scope;
  /** "global" or the channel name whose section provided the value */
  std::string section;
};

/**
 * Settings for one group, keyed by setting name. redev does not interpret
 * backend settings; they are passed to the backend unmodified.
 */
using ResolvedSettings = std::map<std::string, ResolvedSetting>;

/**
 * Layered settings for redev channels and for libraries built on redev.
 *
 * Settings are grouped only by names the user chose: a global section and one
 * section per channel, named path + name from Redev::CreateChannel (in PCMS,
 * the application's path and name, e.g. "core/core"). A channel's settings are
 * its section merged over the global section. Internal ADIOS2 IO names
 * (<path><name>_s2c, <path><name>_c2s) are never exposed.
 *
 * Within a section, settings are organized in named groups. redev uses one
 * group per backend ("adios2.bp4", "adios2.sst", ...) and passes only the
 * group for the channel's transport to the backend. Libraries built on redev
 * (e.g. PCMS) can add their own groups ("pcms.<...>") and read them with Get,
 * sharing the same files, scopes, and unused-setting reporting.
 *
 * The transport is decided by the server: clients receive it from the server
 * during channel setup, and a client's own transport setting is ignored.
 * Transport names are case-insensitive.
 *
 * Invalid configuration is a fatal error: a missing file, a YAML syntax
 * error, an unknown transport, or a non-scalar setting value. Settings are
 * fixed once read: changing a setting after Get/GetSettings has returned it
 * (e.g. after its channel was created) is also a fatal error.
 *
 * Example YAML:
 * \code{.yaml}
 * global:
 *   transport: bp4
 *   adios2.bp4:
 *     BeginStepPollingFrequencySecs: 0.05
 * channels:
 *   core/core:
 *     transport: sst
 *     adios2.sst:
 *       DataTransport: rdma
 * \endcode
 */
class Config {
public:
  class Section;

  /** Builtin scope only. */
  Config();

  /**
   * Merge the file named by REDEV_CONFIG (File scope) and the YAML text in
   * REDEV_CONFIG_YAML (Environment scope) when those are set. Called by the
   * Redev constructors. Collective over comm. Setting either variable in a
   * build without REDEV_ENABLE_YAML is a fatal error.
   */
  void LoadEnvironment(MPI_Comm comm);

  /**
   * Merge a YAML file into the File scope. Rank 0 of comm reads the file and
   * broadcasts its contents. Collective over comm. Requires redev built with
   * REDEV_ENABLE_YAML.
   */
  void LoadFile(const std::string& path, MPI_Comm comm);

  /** Merge YAML text into the given scope (used for REDEV_CONFIG_YAML). */
  void LoadString(const std::string& yaml, ConfigScope scope);

  /** Settings that apply to every channel. */
  Section Global();
  /** Settings that apply only to the channel with this path + name. */
  Section Channel(const std::string& name);

  /**
   * Resolved transport for the named channel as configured on this process.
   * Only meaningful on the server; clients use the server's transport.
   */
  [[nodiscard]] TransportType GetTransport(const std::string& name) const;

  /**
   * Resolved settings of one group for the named section (merged with the
   * global section). Marks the returned settings as used.
   */
  [[nodiscard]] ResolvedSettings Get(const std::string& section,
                                     const std::string& group) const;

  /** Get(name, BackendName(transport)). */
  [[nodiscard]] ResolvedSettings GetSettings(const std::string& name,
                                             TransportType transport) const;

  /** Write each channel's resolved transport and settings with origins. */
  void Print(std::ostream& os) const;

  /**
   * Settings that were never read: keys in groups nobody requested (including
   * backends no channel used), and channel sections for names that were never
   * created (often a misspelled application name). Each entry is
   * "<section>.<group>.<key>".
   */
  [[nodiscard]] std::vector<std::string> Unused() const;

private:
  friend class Section;
  static constexpr std::size_t kNumScopes = 5;
  template <typename T>
  using PerScope = std::array<std::optional<T>, kNumScopes>;
  using Group = std::map<std::string, PerScope<std::string>>;
  struct SectionData {
    PerScope<TransportType> transport;
    std::map<std::string, Group> groups;
  };
  // channel == nullopt refers to the global section
  void SetTransport(const std::optional<std::string>& channel,
                    TransportType transport, ConfigScope scope);
  void SetValue(const std::optional<std::string>& channel,
                const std::string& group, const std::string& key,
                std::string value, ConfigScope scope);
  [[nodiscard]] ResolvedSettings Resolve(const std::string& channel,
                                         const std::string& group) const;
  [[nodiscard]] std::pair<TransportType, ResolvedSetting> ResolveTransport(
    const std::string& channel) const;

  SectionData global_;
  std::map<std::string, SectionData> channels_;
  mutable std::set<std::string> readGlobalGroups_;
  mutable std::set<std::pair<std::string, std::string>> readGroups_;
  mutable bool globalTransportRead_ = false;
  mutable std::set<std::string> readTransports_;
};

class Config::Section {
public:
  Section& SetTransport(TransportType transport,
                        ConfigScope scope = ConfigScope::Override);

  /**
   * Set one setting in a group, e.g.
   * Set("adios2.bp4", "BeginStepPollingFrequencySecs", "0.05").
   */
  Section& Set(const std::string& group, const std::string& key,
               std::string value, ConfigScope scope = ConfigScope::Override);

  /** Set several settings in one group at the same scope. */
  Section& Set(const std::string& group,
               const std::map<std::string, std::string>& settings,
               ConfigScope scope = ConfigScope::Override);

private:
  friend class Config;
  Section(Config& config, std::optional<std::string> channel);
  Config* config_;
  std::optional<std::string> channel_;
};

/** Group name for a transport's backend ("adios2.bp4", "adios2.sst"). */
[[nodiscard]] std::string BackendName(TransportType transport);

/** Transport name ("bp4", "sst"). */
[[nodiscard]] std::string TransportName(TransportType transport);

/** Parse a case-insensitive transport name; unknown names are fatal. */
[[nodiscard]] TransportType ParseTransport(const std::string& name);

} // namespace redev
#endif
